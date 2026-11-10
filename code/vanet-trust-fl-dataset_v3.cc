/* vanet-trust-fl-dataset_v3.cc
 * =============================================================================
 * Trust-Aware VANET Simulation for Federated Learning Dataset Generation (v3)
 * =============================================================================
 *
 * Tested with ns-3.42. Standard WiFi adhoc (802.11a, closest to 802.11p) + AODV.
 *
 * Changes relative to v2:
 *
 *   [A1] REAL PACKET DROPPING. In v2 the attacker's "drops" were counters
 *        incremented inside a PHY trace callback; AODV still forwarded every
 *        packet, so attackers had no effect on the network. v3 installs a
 *        SelectiveDropRouting protocol at higher priority than AODV inside an
 *        Ipv4ListRouting stack. On attacker nodes it consumes transit data
 *        packets with probability dropRate before AODV can forward them.
 *        Attackers still run AODV normally (they answer and relay RREQ/RREP),
 *        so this is a selective-forwarding / grayhole attacker
 *        (dropRate = 1.0 gives a pure data-plane blackhole without route
 *        attraction; fake-RREP attraction would need a patched AODV).
 *        Honest nodes get the same layer with dropRate = 0, so the routing
 *        stacks are identical apart from the drop probability.
 *
 *   [A1] OBSERVABLE FEATURES FROM A WATCHDOG. forward_ratio is no longer
 *        computed from the simulator's own bookkeeping. When node A hands a
 *        data frame to relay X (MAC addr1 = X, IP dst != X), the watchdog
 *        opens an entry. If A's PHY later decodes X re-transmitting the same
 *        packet (promiscuous overhearing, addr2 = X), the forward counts as
 *        overheard. Every entry closes after wdTimeout seconds:
 *            fwd_requested++ ; fwd_overheard++ if A overheard it.
 *        A real watchdog cannot tell "X dropped it" from "X forwarded it but
 *        A was out of range / collided", so neither can this one. Those
 *        mobility- and collision-induced false accusations are what make
 *        honest forward_ratio < 1. Ground-truth splits are written ONLY to
 *        the summary file, never to the feature CSV.
 *
 *   [A1] PHY reception failures are no longer counted as "drops" (that is
 *        why v2 honest forward_ratio was ~0.25). route_participation is now
 *        unique RREQs (origin, id) rebroadcast / unique RREQs received.
 *
 *   Mobility: --mobility ring (default) puts the 2 km road on a closed loop so
 *   the network does not partition; --mobility highway is the v2 open road.
 *
 *   On-off attackers: --onTime T1 --offTime T2 drop only during T1 of every
 *   T1+T2 cycle (random phase per attacker), to rebuild trust in between.
 *
 *   Attackers are a seeded random subset (--placement random). The v2 rule
 *   (--placement spaced) puts all attackers in one block for ratios >= 0.5.
 *
 *   [A4] NEIGHBOUR LOG. <prefix>_neighbors.csv lists, every second, which
 *        vehicles are within range of which. The FL pipeline uses it to give
 *        each vehicle a local dataset made of what it actually observed.
 *        Positions (pos_x, pos_y) are also exported (not used as features).
 *
 * Outputs:
 *   <prefix>_features.csv   one row per (second, vehicle), cumulative counters
 *   <prefix>_neighbors.csv  timestamp, vehicle_id, neighbor_id
 *   <prefix>_summary.txt    parameters, PDR, ground-truth drop/watchdog stats
 *
 * Usage:
 *   ./ns3 run "vanet-trust-fl-dataset_v3 --nodes=50 --malicious=0.3 --dropRate=0.5 --seed=1"
 */

#include "ns3/aodv-module.h"
#include "ns3/aodv-packet.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/llc-snap-header.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/udp-header.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VanetTrustFLDatasetV3");

static const uint16_t AODV_PORT = 654;

// =============================================================================
// GLOBAL STATE
// =============================================================================

struct VehicleStats
{
    uint32_t vehicleId = 0;
    uint64_t packetsSent = 0;
    uint64_t packetsReceived = 0;
    uint64_t bytesSent = 0;
    uint64_t bytesReceived = 0;
    uint64_t dataPktsSent = 0;     // data frames this node transmitted
    uint64_t dataPktsReceived = 0; // data frames addressed to this node
    uint64_t rreqReceived = 0;     // unique (origin, id), not originated here
    uint64_t rreqForwarded = 0;    // unique (origin, id) rebroadcast
    uint64_t rrepSent = 0;
    uint64_t rrepReceived = 0;
    // watchdog view of this node as a relay (observable)
    uint64_t fwdRequested = 0;
    uint64_t fwdOverheard = 0;
    // ground truth, summary file only
    uint64_t wdTrulyForwarded = 0;
    uint64_t wdNotForwarded = 0;
    uint64_t trueDrops = 0;        // packets consumed by SelectiveDropRouting
    uint32_t neighborCount = 0;
    double totalDelay = 0;
    uint32_t delayCount = 0;
    bool isMalicious = false;
};

std::map<uint32_t, VehicleStats> g_vehicleStats;
std::set<uint32_t> g_blackholeNodes;
std::map<uint64_t, Time> g_packetSendTime;

std::map<Mac48Address, uint32_t> g_macToNode;
std::map<uint32_t, uint32_t> g_ipToNode; // Ipv4Address::Get() -> node id
std::vector<Ipv4Address> g_nodeIp;

typedef std::pair<uint32_t, uint32_t> RreqKey; // (origin ip, rreq id)
std::map<uint32_t, std::set<RreqKey>> g_rreqSeen;
std::map<uint32_t, std::set<RreqKey>> g_rreqRelayed;

struct WdEntry
{
    uint32_t prevHop;
    bool forwarded = false;
    bool overheard = false;
};

typedef std::pair<uint64_t, uint32_t> WdKey; // (packet uid, relay node)
std::map<WdKey, WdEntry> g_wd;
std::set<WdKey> g_wdDone;

uint32_t g_numNodes = 50;
double g_maliciousRatio = 0.2;
double g_dropRate = 0.9;
double g_simDuration = 100.0;
double g_wdTimeout = 1.0;

// =============================================================================
// SELECTIVE-DROP ROUTING LAYER (A1)
// =============================================================================
// Sits above AODV in an Ipv4ListRouting stack. Ipv4ListRouting delivers
// local/broadcast traffic before asking any protocol, so this layer only ever
// sees transit unicast packets. Returning true without calling a callback
// silently discards the packet. Returning false hands it to AODV.

class SelectiveDropRouting : public Ipv4RoutingProtocol
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::SelectiveDropRouting")
                                .SetParent<Ipv4RoutingProtocol>()
                                .SetGroupName("Internet")
                                .AddConstructor<SelectiveDropRouting>();
        return tid;
    }

    SelectiveDropRouting()
    {
        m_rng = CreateObject<UniformRandomVariable>();
    }

    void Configure(double dropRate, uint32_t nodeId, double onTime = 0.0, double offTime = 0.0,
                   double phase = 0.0)
    {
        m_dropRate = dropRate;
        m_nodeId = nodeId;
        m_onTime = onTime;
        m_offTime = offTime;
        m_phase = phase;
    }

    Ptr<Ipv4Route> RouteOutput(Ptr<Packet>,
                               const Ipv4Header&,
                               Ptr<NetDevice>,
                               Socket::SocketErrno& sockerr) override
    {
        sockerr = Socket::ERROR_NOROUTETOHOST;
        return nullptr; // never originates routes; AODV does
    }

    bool RouteInput(Ptr<const Packet> p,
                    const Ipv4Header& header,
                    Ptr<const NetDevice>,
                    const UnicastForwardCallback&,
                    const MulticastForwardCallback&,
                    const LocalDeliverCallback&,
                    const ErrorCallback&) override
    {
        if (!m_ipv4 || m_dropRate <= 0.0)
        {
            return false;
        }
        // Own packets coming back through loopback (AODV deferred route).
        if (m_ipv4->GetInterfaceForAddress(header.GetSource()) >= 0)
        {
            return false;
        }
        Ipv4Address dst = header.GetDestination();
        if (dst.IsBroadcast() || dst.IsMulticast())
        {
            return false;
        }
        if (header.GetProtocol() == 17)
        {
            UdpHeader udp;
            p->PeekHeader(udp);
            if (udp.GetDestinationPort() == AODV_PORT || udp.GetSourcePort() == AODV_PORT)
            {
                return false; // control plane untouched
            }
        }
        // On-off attacker: drop only during the "on" part of each on+off cycle
        // (own random phase), behave honestly otherwise.
        if (m_onTime > 0.0)
        {
            double t = Simulator::Now().GetSeconds() + m_phase;
            if (std::fmod(t, m_onTime + m_offTime) >= m_onTime)
            {
                return false;
            }
        }
        if (m_rng->GetValue() < m_dropRate)
        {
            g_vehicleStats[m_nodeId].trueDrops++;
            return true; // consumed: never forwarded
        }
        return false;
    }

    void NotifyInterfaceUp(uint32_t) override {}
    void NotifyInterfaceDown(uint32_t) override {}
    void NotifyAddAddress(uint32_t, Ipv4InterfaceAddress) override {}
    void NotifyRemoveAddress(uint32_t, Ipv4InterfaceAddress) override {}

    void SetIpv4(Ptr<Ipv4> ipv4) override
    {
        m_ipv4 = ipv4;
    }

    void PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit) const override
    {
        *stream->GetStream() << "SelectiveDropRouting dropRate=" << m_dropRate << "\n";
    }

  private:
    Ptr<Ipv4> m_ipv4;
    Ptr<UniformRandomVariable> m_rng;
    double m_dropRate = 0.0;
    double m_onTime = 0.0;
    double m_offTime = 0.0;
    double m_phase = 0.0;
    uint32_t m_nodeId = 0;
};

NS_OBJECT_ENSURE_REGISTERED(SelectiveDropRouting);

// =============================================================================
// FRAME PARSING
// =============================================================================

struct FrameInfo
{
    bool isData = false;  // MAC data frame carrying IPv4
    bool isAodv = false;
    bool hasIp = false;
    Mac48Address addr1, addr2;
    Ipv4Address ipSrc, ipDst;
    uint8_t aodvType = 0;
    RreqKey rreq{0, 0};
};

FrameInfo ParseFrame(Ptr<const Packet> packet)
{
    FrameInfo fi;
    Ptr<Packet> p = packet->Copy();
    WifiMacHeader mac;
    if (p->GetSize() < 4 || p->PeekHeader(mac) == 0)
    {
        return fi;
    }
    p->RemoveHeader(mac);
    if (!mac.IsData())
    {
        return fi;
    }
    fi.addr1 = mac.GetAddr1();
    fi.addr2 = mac.GetAddr2();

    LlcSnapHeader llc;
    if (p->GetSize() < llc.GetSerializedSize())
    {
        return fi;
    }
    p->RemoveHeader(llc);
    if (llc.GetType() != 0x0800)
    {
        return fi;
    }
    Ipv4Header ip;
    if (p->GetSize() < ip.GetSerializedSize())
    {
        return fi;
    }
    p->RemoveHeader(ip);
    fi.hasIp = true;
    fi.ipSrc = ip.GetSource();
    fi.ipDst = ip.GetDestination();
    fi.isData = true;

    if (ip.GetProtocol() != 17)
    {
        return fi;
    }
    UdpHeader udp;
    if (p->GetSize() < udp.GetSerializedSize())
    {
        return fi;
    }
    p->RemoveHeader(udp);
    if (udp.GetDestinationPort() != AODV_PORT)
    {
        return fi;
    }

    fi.isData = false;
    fi.isAodv = true;
    aodv::TypeHeader th;
    if (p->GetSize() < th.GetSerializedSize())
    {
        return fi;
    }
    p->RemoveHeader(th);
    if (!th.IsValid())
    {
        return fi;
    }
    fi.aodvType = th.Get();
    if (th.Get() == aodv::AODVTYPE_RREQ)
    {
        aodv::RreqHeader rh;
        if (p->GetSize() >= rh.GetSerializedSize())
        {
            p->RemoveHeader(rh);
            fi.rreq = RreqKey(rh.GetOrigin().Get(), rh.GetId());
        }
    }
    return fi;
}

uint32_t NodeFromContext(const std::string& context)
{
    std::size_t pos1 = context.find("/NodeList/") + 10;
    std::size_t pos2 = context.find("/", pos1);
    return std::stoi(context.substr(pos1, pos2 - pos1));
}

// =============================================================================
// WATCHDOG
// =============================================================================

void WatchdogTimeout(WdKey key)
{
    auto it = g_wd.find(key);
    if (it == g_wd.end())
    {
        return;
    }
    VehicleStats& relay = g_vehicleStats[key.second];
    relay.fwdRequested++;
    if (it->second.overheard)
    {
        relay.fwdOverheard++;
    }
    if (it->second.forwarded)
    {
        relay.wdTrulyForwarded++;
    }
    else
    {
        relay.wdNotForwarded++;
    }
    g_wd.erase(it);
    g_wdDone.insert(key);
}

// =============================================================================
// TRACE CALLBACKS
// =============================================================================

void PhyTxBeginTrace(std::string context, Ptr<const Packet> packet, double)
{
    uint32_t a = NodeFromContext(context);
    VehicleStats& st = g_vehicleStats[a];
    st.packetsSent++;
    st.bytesSent += packet->GetSize();
    g_packetSendTime[packet->GetUid()] = Simulator::Now();

    FrameInfo fi = ParseFrame(packet);

    if (fi.isAodv)
    {
        if (fi.aodvType == aodv::AODVTYPE_RREQ && fi.rreq.first != g_nodeIp[a].Get())
        {
            if (g_rreqSeen[a].count(fi.rreq) && !g_rreqRelayed[a].count(fi.rreq))
            {
                g_rreqRelayed[a].insert(fi.rreq);
                st.rreqForwarded++;
            }
        }
        else if (fi.aodvType == aodv::AODVTYPE_RREP)
        {
            st.rrepSent++;
        }
        return;
    }
    if (!fi.isData)
    {
        return;
    }
    st.dataPktsSent++;
    uint64_t uid = packet->GetUid();

    // A is (re)transmitting a packet it was asked to relay -> forwarded.
    auto self = g_wd.find(WdKey(uid, a));
    if (self != g_wd.end())
    {
        self->second.forwarded = true;
    }

    // A hands the packet to relay X: open a watchdog entry, unless X is the
    // final destination (nothing to forward) or it is a MAC retry.
    if (fi.addr1.IsGroup())
    {
        return;
    }
    auto rx = g_macToNode.find(fi.addr1);
    if (rx == g_macToNode.end())
    {
        return;
    }
    uint32_t relay = rx->second;
    auto dn = g_ipToNode.find(fi.ipDst.Get());
    if (dn != g_ipToNode.end() && dn->second == relay)
    {
        return;
    }
    WdKey key(uid, relay);
    if (g_wd.count(key) || g_wdDone.count(key))
    {
        return;
    }
    WdEntry e;
    e.prevHop = a;
    g_wd[key] = e;
    Simulator::Schedule(Seconds(g_wdTimeout), &WatchdogTimeout, key);
}

void PhyRxEndTrace(std::string context, Ptr<const Packet> packet)
{
    uint32_t o = NodeFromContext(context);
    VehicleStats& st = g_vehicleStats[o];
    st.packetsReceived++;
    st.bytesReceived += packet->GetSize();

    uint64_t uid = packet->GetUid();
    auto ts = g_packetSendTime.find(uid);
    if (ts != g_packetSendTime.end())
    {
        st.totalDelay += (Simulator::Now() - ts->second).GetMilliSeconds();
        st.delayCount++;
    }

    FrameInfo fi = ParseFrame(packet);
    if (!fi.hasIp)
    {
        return;
    }
    bool toMe = (g_macToNode.count(fi.addr1) && g_macToNode[fi.addr1] == o);

    if (fi.isAodv)
    {
        if (fi.aodvType == aodv::AODVTYPE_RREQ && fi.rreq.first != g_nodeIp[o].Get())
        {
            if (g_rreqSeen[o].insert(fi.rreq).second)
            {
                st.rreqReceived++;
            }
        }
        else if (fi.aodvType == aodv::AODVTYPE_RREP && toMe)
        {
            st.rrepReceived++;
        }
        return;
    }
    if (toMe)
    {
        st.dataPktsReceived++;
    }

    // Promiscuous overhearing: O decoded relay T re-transmitting a packet
    // that O itself handed to T.
    auto tx = g_macToNode.find(fi.addr2);
    if (tx == g_macToNode.end())
    {
        return;
    }
    auto it = g_wd.find(WdKey(uid, tx->second));
    if (it != g_wd.end() && it->second.prevHop == o)
    {
        it->second.overheard = true;
    }
}

// =============================================================================
// SNAPSHOTS
// =============================================================================

std::ofstream g_featureFile;
std::ofstream g_neighborFile;

void OpenOutputFiles(const std::string& prefix)
{
    g_featureFile.open(prefix + "_features.csv");
    g_featureFile << "timestamp,vehicle_id,pos_x,pos_y,"
                  << "packets_sent,packets_received,bytes_sent,bytes_received,"
                  << "data_pkts_sent,data_pkts_received,"
                  << "rreq_received,rreq_forwarded,rrep_sent,rrep_received,"
                  << "fwd_requested,fwd_overheard,"
                  << "forward_ratio,route_participation,"
                  << "avg_delay_ms,neighbor_count,label\n";
    g_neighborFile.open(prefix + "_neighbors.csv");
    g_neighborFile << "timestamp,vehicle_id,neighbor_id\n";
}

void SnapshotFeatures(NodeContainer nodes, double range)
{
    double now = Simulator::Now().GetSeconds();
    uint32_t n = nodes.GetN();
    std::vector<Vector> pos(n);
    for (uint32_t i = 0; i < n; i++)
    {
        pos[i] = nodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
    }
    for (uint32_t i = 0; i < n; i++)
    {
        uint32_t count = 0;
        for (uint32_t j = 0; j < n; j++)
        {
            if (i != j && CalculateDistance(pos[i], pos[j]) <= range)
            {
                count++;
                g_neighborFile << std::fixed << std::setprecision(1) << now << "," << i << ","
                               << j << "\n";
            }
        }
        g_vehicleStats[i].neighborCount = count;
    }

    for (uint32_t i = 0; i < n; i++)
    {
        VehicleStats& s = g_vehicleStats[i];
        double forwardRatio =
            (s.fwdRequested > 0) ? static_cast<double>(s.fwdOverheard) / s.fwdRequested : 0.0;
        double routeRatio =
            (s.rreqReceived > 0)
                ? std::min(1.0, static_cast<double>(s.rreqForwarded) / s.rreqReceived)
                : 0.0;
        double avgDelay = (s.delayCount > 0) ? s.totalDelay / s.delayCount : 0.0;

        g_featureFile << std::fixed << std::setprecision(4) << now << "," << i << ","
                      << std::setprecision(1) << pos[i].x << "," << pos[i].y << ","
                      << s.packetsSent << "," << s.packetsReceived << "," << s.bytesSent << ","
                      << s.bytesReceived << "," << s.dataPktsSent << "," << s.dataPktsReceived
                      << "," << s.rreqReceived << "," << s.rreqForwarded << "," << s.rrepSent
                      << "," << s.rrepReceived << "," << s.fwdRequested << "," << s.fwdOverheard
                      << "," << std::setprecision(4) << forwardRatio << "," << routeRatio << ","
                      << std::setprecision(2) << avgDelay << "," << s.neighborCount << ","
                      << (s.isMalicious ? 1 : 0) << "\n";
    }
}

// =============================================================================
// TRAFFIC
// =============================================================================

void SetupTrafficFlows(NodeContainer& nodes,
                       Ipv4InterfaceContainer& interfaces,
                       uint32_t numFlows,
                       double duration)
{
    uint16_t basePort = 9000;
    uint32_t packetSize = 512;

    Ptr<UniformRandomVariable> randNode = CreateObject<UniformRandomVariable>();
    randNode->SetAttribute("Min", DoubleValue(0));
    randNode->SetAttribute("Max", DoubleValue(nodes.GetN() - 1));

    for (uint32_t i = 0; i < numFlows; i++)
    {
        uint32_t src = randNode->GetInteger();
        uint32_t dst = randNode->GetInteger();
        while (dst == src)
        {
            dst = randNode->GetInteger();
        }
        uint16_t port = basePort + i;

        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(nodes.Get(dst));
        sinkApp.Start(Seconds(0.5));
        sinkApp.Stop(Seconds(duration - 0.5));

        OnOffHelper source("ns3::UdpSocketFactory",
                           InetSocketAddress(interfaces.GetAddress(dst), port));
        source.SetAttribute("PacketSize", UintegerValue(packetSize));
        source.SetAttribute("DataRate", DataRateValue(DataRate("50Kbps")));
        source.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        source.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer sourceApp = source.Install(nodes.Get(src));

        double startTime = 1.0 + (i * 0.1);
        if (startTime < duration - 2.0)
        {
            sourceApp.Start(Seconds(startTime));
            sourceApp.Stop(Seconds(duration - 1.0));
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    uint32_t numNodes = 50;
    double duration = 100.0;
    double maliciousRatio = 0.2;
    double dropRate = 0.9;
    uint32_t numFlows = 30;  // 100 saturates the 6 Mbps channel (honest fwd_ratio 0.42)
    double commRange = 250.0;
    double wdTimeout = 1.0;
    double onTime = 0.0;  // 0 = attackers drop all the time
    double offTime = 0.0;
    std::string outputPrefix = "vanet_fl_v3";
    std::string placement = "random";
    std::string mobilityModel = "ring";
    std::string propagation = "range";
    uint32_t seed = 42;

    CommandLine cmd;
    cmd.AddValue("nodes", "Number of vehicles", numNodes);
    cmd.AddValue("duration", "Simulation duration (seconds)", duration);
    cmd.AddValue("malicious", "Ratio of malicious vehicles (0.0-1.0)", maliciousRatio);
    cmd.AddValue("dropRate",
                 "Probability an attacker discards a transit data packet. "
                 "1.0 = data-plane blackhole, 0.3-0.6 = grayhole",
                 dropRate);
    cmd.AddValue("flows", "Number of traffic flows", numFlows);
    cmd.AddValue("range", "Communication range (meters)", commRange);
    cmd.AddValue("onTime", "On-off attacker: seconds of dropping per cycle (0 = always on)", onTime);
    cmd.AddValue("offTime", "On-off attacker: seconds of honest forwarding per cycle", offTime);
    cmd.AddValue("wdTimeout", "Watchdog wait before a relay counts as not forwarding (s)",
                 wdTimeout);
    cmd.AddValue("mobility", "ring (closed 2 km loop, default) or highway (v2 open road)",
                 mobilityModel);
    cmd.AddValue("propagation", "range (unit disk, default) or logdist (v2: LogDistance + cutoff)",
                 propagation);
    cmd.AddValue("placement", "Attacker selection: random (seeded) or spaced (v2 rule)", placement);
    cmd.AddValue("output", "Output filename prefix", outputPrefix);
    cmd.AddValue("seed", "Random seed", seed);
    cmd.Parse(argc, argv);

    g_numNodes = numNodes;
    g_maliciousRatio = maliciousRatio;
    g_dropRate = dropRate;
    g_simDuration = duration;
    g_wdTimeout = wdTimeout;

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(1);

    // ---- attackers ----
    // random (default): a seeded random subset. spaced: v2 rule. The v2 rule
    // uses step = N / (k + 1), which collapses to 1 for ratios >= 0.5 and puts
    // every attacker in one contiguous block of ids, i.e. one stretch of road;
    // honest vehicles in the other half then never observe an attacker.
    uint32_t numMalicious = static_cast<uint32_t>(numNodes * maliciousRatio);
    if (numMalicious > 0 && placement == "random")
    {
        std::vector<uint32_t> ids(numNodes);
        for (uint32_t i = 0; i < numNodes; i++)
        {
            ids[i] = i;
        }
        Ptr<UniformRandomVariable> pick = CreateObject<UniformRandomVariable>();
        for (uint32_t i = numNodes - 1; i > 0; i--)
        {
            std::swap(ids[i], ids[pick->GetInteger(0, i)]);
        }
        g_blackholeNodes.insert(ids.begin(), ids.begin() + numMalicious);
    }
    else if (numMalicious > 0)
    {
        uint32_t step = numNodes / (numMalicious + 1);
        for (uint32_t i = 0; i < numMalicious; i++)
        {
            uint32_t id = (i + 1) * step;
            if (id < numNodes)
            {
                g_blackholeNodes.insert(id);
            }
        }
    }

    NodeContainer nodes;
    nodes.Create(numNodes);
    for (uint32_t i = 0; i < numNodes; i++)
    {
        g_vehicleStats[i].vehicleId = i;
        g_vehicleStats[i].isMalicious = g_blackholeNodes.count(i) > 0;
    }

    // ---- WiFi ----
    // range (default): unit disk, every frame within commRange is heard (subject
    //   to collisions). Matches the neighbour log and the watchdog's notion of range.
    // logdist: v2 channel = ns-3 default LogDistance loss + range cutoff; at 20 dBm
    //   the real decode range is far below commRange.
    YansWifiChannelHelper wifiChannel;
    if (propagation == "logdist")
    {
        wifiChannel = YansWifiChannelHelper::Default();
    }
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel",
                                   "MaxRange",
                                   DoubleValue(commRange));
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(20.0));
    wifiPhy.Set("TxPowerEnd", DoubleValue(20.0));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("OfdmRate6Mbps"),
                                 "ControlMode",
                                 StringValue("OfdmRate6Mbps"));
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);

    // ---- mobility: 2 km road, 2 lanes, 20-30 m/s ----
    // ring (default): the road is a closed loop (circumference roadLength), so
    //   density stays constant; faster cars lap slower ones and the topology
    //   keeps changing. Positions are waypoints every 0.5 s along the circle
    //   (chord error < 0.1 m).
    // highway: v2 open road; cars with different speeds drift apart and the
    //   network partitions (app PDR about 0.05-0.13).
    double roadLength = 2000.0;
    double laneWidth = 4.0;
    Ptr<UniformRandomVariable> speedRand = CreateObject<UniformRandomVariable>();
    speedRand->SetAttribute("Min", DoubleValue(20.0));
    speedRand->SetAttribute("Max", DoubleValue(30.0));
    MobilityHelper mobility;
    if (mobilityModel == "ring")
    {
        mobility.SetMobilityModel("ns3::WaypointMobilityModel");
        mobility.Install(nodes);
        double r0 = roadLength / (2 * M_PI);
        for (uint32_t i = 0; i < numNodes; i++)
        {
            double r = r0 + ((i % 2 == 0) ? 0.0 : laneWidth);
            double s0 = (roadLength / numNodes) * i;
            double v = speedRand->GetValue();
            Ptr<WaypointMobilityModel> wp = nodes.Get(i)->GetObject<WaypointMobilityModel>();
            for (double t = 0.0; t <= duration + 0.5; t += 0.5)
            {
                double th = (s0 + v * t) / r0; // same angular progress on both lanes
                wp->AddWaypoint(Waypoint(Seconds(t), Vector(r * std::cos(th), r * std::sin(th), 1.5)));
            }
        }
    }
    else
    {
        Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
        for (uint32_t i = 0; i < numNodes; i++)
        {
            posAlloc->Add(Vector((roadLength / numNodes) * i, (i % 2 == 0) ? 0.0 : laneWidth, 1.5));
        }
        mobility.SetPositionAllocator(posAlloc);
        mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        mobility.Install(nodes);
        for (uint32_t i = 0; i < numNodes; i++)
        {
            nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
                Vector(speedRand->GetValue(), 0.0, 0.0));
        }
    }

    // ---- routing: Ipv4ListRouting { SelectiveDropRouting (10), AODV (0) } ----
    AodvHelper aodv;
    aodv.Set("EnableHello", BooleanValue(true));
    aodv.Set("HelloInterval", TimeValue(Seconds(1.0)));
    Ipv4ListRoutingHelper list;
    list.Add(aodv, 0);

    InternetStackHelper internet;
    internet.SetRoutingHelper(list);
    internet.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.0.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Created only for on-off runs: an extra random variable would shift ns-3's
    // stream assignment and change every default dataset.
    Ptr<UniformRandomVariable> phaseRand =
        (onTime > 0.0) ? CreateObject<UniformRandomVariable>() : nullptr;
    g_nodeIp.resize(numNodes);
    for (uint32_t i = 0; i < numNodes; i++)
    {
        g_nodeIp[i] = interfaces.GetAddress(i);
        g_ipToNode[g_nodeIp[i].Get()] = i;
        g_macToNode[Mac48Address::ConvertFrom(devices.Get(i)->GetAddress())] = i;

        Ptr<Ipv4ListRouting> lr =
            DynamicCast<Ipv4ListRouting>(nodes.Get(i)->GetObject<Ipv4>()->GetRoutingProtocol());
        NS_ABORT_MSG_IF(!lr, "expected Ipv4ListRouting");
        Ptr<SelectiveDropRouting> drop = CreateObject<SelectiveDropRouting>();
        if (g_blackholeNodes.count(i) && onTime > 0.0)
        {
            drop->Configure(dropRate, i, onTime, offTime, phaseRand->GetValue(0.0, onTime + offTime));
        }
        else
        {
            drop->Configure(g_blackholeNodes.count(i) ? dropRate : 0.0, i);
        }
        lr->AddRoutingProtocol(drop, 10);
    }

    SetupTrafficFlows(nodes, interfaces, numFlows, duration);

    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                    MakeCallback(&PhyTxBeginTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxEnd",
                    MakeCallback(&PhyRxEndTrace));

    FlowMonitorHelper fmHelper;
    Ptr<FlowMonitor> flowmon = fmHelper.InstallAll();

    OpenOutputFiles(outputPrefix);
    for (double t = 1.0; t <= duration - 2.0; t += 1.0)
    {
        Simulator::Schedule(Seconds(t), &SnapshotFeatures, nodes, commRange);
    }

    Simulator::Stop(Seconds(duration));
    Simulator::Run();

    g_featureFile.close();
    g_neighborFile.close();

    // ---- packet delivery ratio of the application flows ----
    flowmon->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> cls = DynamicCast<Ipv4FlowClassifier>(fmHelper.GetClassifier());
    uint64_t txApp = 0;
    uint64_t rxApp = 0;
    for (auto& kv : flowmon->GetFlowStats())
    {
        Ipv4FlowClassifier::FiveTuple t = cls->FindFlow(kv.first);
        if (t.destinationPort >= 9000)
        {
            txApp += kv.second.txPackets;
            rxApp += kv.second.rxPackets;
        }
    }
    double pdr = txApp ? static_cast<double>(rxApp) / txApp : 0.0;

    // ---- ground-truth summary (never fed to the FL pipeline as features) ----
    double frH = 0, frM = 0, nfH = 0, nfM = 0;
    uint32_t nH = 0, nM = 0;
    uint64_t drops = 0, req = 0, missed = 0;
    for (auto& kv : g_vehicleStats)
    {
        VehicleStats& v = kv.second;
        drops += v.trueDrops;
        req += v.fwdRequested;
        if (v.fwdRequested == 0)
        {
            continue;
        }
        double fr = static_cast<double>(v.fwdOverheard) / v.fwdRequested;
        double nf = static_cast<double>(v.wdNotForwarded) / v.fwdRequested;
        missed += v.wdTrulyForwarded - std::min(v.wdTrulyForwarded, v.fwdOverheard);
        if (v.isMalicious)
        {
            frM += fr;
            nfM += nf;
            nM++;
        }
        else
        {
            frH += fr;
            nfH += nf;
            nH++;
        }
    }
    if (nH)
    {
        frH /= nH;
        nfH /= nH;
    }
    if (nM)
    {
        frM /= nM;
        nfM /= nM;
    }

    std::ofstream f(outputPrefix + "_summary.txt");
    f << "Nodes: " << g_numNodes << "\n"
      << "Malicious ratio: " << (g_maliciousRatio * 100) << "%\n"
      << "Malicious count: " << g_blackholeNodes.size() << "\n"
      << "Drop rate: " << g_dropRate << "\n"
      << "On/off (s): " << onTime << "/" << offTime << "\n"
      << "Duration: " << g_simDuration << "s\n"
      << "Seed: " << seed << "\n"
      << "Placement: " << placement << "\n"
      << "Mobility: " << mobilityModel << "\n"
      << "Propagation: " << propagation << "\n"
      << "Watchdog timeout: " << g_wdTimeout << "s\n"
      << "App PDR: " << std::fixed << std::setprecision(4) << pdr << " (" << rxApp << "/"
      << txApp << ")\n"
      << "Packets dropped by attackers (ground truth): " << drops << "\n"
      << "Watchdog relay requests: " << req << "\n"
      << "Watchdog false accusations (forwarded, not overheard): " << missed << "\n"
      << "Mean watchdog forward_ratio honest: " << frH << " malicious: " << frM << "\n"
      << "Mean truly-not-forwarded rate honest: " << nfH << " malicious: " << nfM << "\n"
      << "Attacker IDs:";
    for (auto id : g_blackholeNodes)
    {
        f << " " << id;
    }
    f << "\n";
    f.close();

    std::cout << "\n=== " << outputPrefix << " ===\n"
              << "  nodes " << g_numNodes << " | malicious " << g_blackholeNodes.size()
              << " | dropRate " << g_dropRate << " | PDR " << std::setprecision(3) << pdr << "\n"
              << "  attacker drops " << drops << " | watchdog fwd_ratio honest " << frH
              << " malicious " << frM << "\n"
              << "  wrote " << outputPrefix << "_features.csv, _neighbors.csv, _summary.txt\n";

    Simulator::Destroy();
    return 0;
}
