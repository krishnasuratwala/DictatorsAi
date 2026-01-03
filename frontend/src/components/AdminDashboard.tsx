import React, { useState, useEffect } from 'react';
import { db, User, StoredSession } from '../services/db';
import { XMarkIcon, EyeIcon, UserGroupIcon, CurrencyDollarIcon, ShieldCheckIcon, HandThumbUpIcon, HandThumbDownIcon, CommandLineIcon } from '@heroicons/react/24/solid';

interface AdminDashboardProps {
    onClose: () => void;
}

const AdminDashboard: React.FC<AdminDashboardProps> = ({ onClose }) => {
    const [stats, setStats] = useState<any>(null);
    const [users, setUsers] = useState<User[]>([]);
    const [selectedUser, setSelectedUser] = useState<User | null>(null);
    const [userChats, setUserChats] = useState<StoredSession[]>([]);
    const [payouts, setPayouts] = useState<any[]>([]);
    const [donations, setDonations] = useState<any[]>([]);
    const [transactions, setTransactions] = useState<any[]>([]); // New State
    const [activeTab, setActiveTab] = useState<'activity' | 'commissions' | 'donations' | 'financials' | 'logs'>('activity');
    const [loading, setLoading] = useState(true);

    // Logs State
    const [logs, setLogs] = useState<string[]>([]);
    const [logSource, setLogSource] = useState<'gpu' | 'llm' | 'tunnel'>('gpu');
    const [isLogStreaming, setIsLogStreaming] = useState(false);

    useEffect(() => {
        loadData();
    }, []);

    // --- LOG STREAMING EFFECT ---
    useEffect(() => {
        let eventSource: EventSource | null = null;

        if (activeTab === 'logs') {
            setLogs([]); // Clear on switch
            setIsLogStreaming(true);
            const token = localStorage.getItem('dictator_token');
            const baseUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

            // Use EventSourcePolyfill if headers needed, or pass via URL param if backend supports it.
            // But standard EventSource doesn't support headers easily.
            // Alternative: Fetch loop or use library. 
            // SIMPLIFICATION: We'll assume the backend checks cookie or we pass token in URL.
            // Since we implemented 'token_required', we need to pass it.
            // Standard EventSource can't pass Authorization header.
            // FIX: We will rely on a fetching loop for now OR use a library.
            // ACTUALLY: Let's use `fetch` with a `ReadableStream` reader (like the chat) instead of EventSource
            // because we already have that logic working for Chat! 

            const startStream = async () => {
                try {
                    const response = await fetch(`${baseUrl}/api/admin/logs?source=${logSource}`, {
                        headers: { 'Authorization': `Bearer ${token}` }
                    });

                    if (!response.body) return;
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value);
                        // Parse SSE lines "data: ..."
                        const lines = chunk.split('\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const text = line.slice(6);
                                setLogs(prev => [...prev.slice(-199), text]); // Keep last 200 lines
                            }
                        }
                    }
                } catch (e) {
                    console.error("Log Stream Error", e);
                    setLogs(prev => [...prev, "--- CONNECTION INTERRUPTED ---"]);
                } finally {
                    setIsLogStreaming(false);
                }
            };

            startStream();

            return () => {
                // Ideally cancel fetch, but simple implementation: just let it die or component unmount
                // In a real app we'd use AbortController
            };
        }
    }, [activeTab, logSource]);

    const loadData = async () => {
        setLoading(true);
        try {
            const [statsData, usersData, payoutsData, donationsData, transactionsData] = await Promise.all([
                db.getAdminStats(),
                db.getAdminUsers(),
                db.getAdminPayouts(),
                db.getAdminDonations(),
                db.getAdminTransactions()
            ]);
            setStats(statsData);
            setUsers(usersData);
            setPayouts(payoutsData.filter((p: any) => p.status === 'pending'));
            setDonations(donationsData);
            setTransactions(transactionsData);
        } catch (e) {
            console.error("Admin Load Error", e);
        } finally {
            setLoading(false);
        }
    };

    const handleInspectUser = async (user: User) => {
        setSelectedUser(user);
        try {
            const chats = await db.getAdminUserChats(user.id);
            setUserChats(chats);
        } catch (e) {
            console.error("Chat Load Error", e);
        }
    };

    const handleDeleteUser = async (e: React.MouseEvent, userId: string) => {
        e.stopPropagation(); // Prevent opening inspector
        if (!window.confirm("CONFIRM EXECUTION: Permanently delete this agent and all associated archives? This action cannot be undone.")) return;

        try {
            await db.deleteAdminUser(userId);
            // Quick update local list
            setUsers(prev => prev.filter(u => u.id !== userId));
            if (selectedUser?.id === userId) {
                setSelectedUser(null);
                setUserChats([]);
            }
        } catch (err) {
            console.error("Deletion failed", err);
            alert("Execution Failed");
        }
    };

    const handleConfirmPayout = async (payoutId: string) => {
        const proof = window.prompt("Enter Transaction Hash / Proof of Payment:");
        if (!proof) return;

        try {
            await db.confirmPayout(payoutId, proof);
            // Remove from list
            setPayouts(prev => prev.filter(p => p.id !== payoutId));
            alert("Payment Confirmed");
        } catch (e) {
            console.error(e);
            alert("Failed to confirm payment");
        }
    };

    return (
        <div className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-md flex flex-col p-4 md:p-8 animate-in fade-in duration-300 overflow-y-auto md:overflow-hidden">

            {/* Header ... */}
            <div className="flex items-center justify-between mb-8 shrink-0 border-b border-zinc-800 pb-4">
                {/* ... existing header ... */}
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-red-950/30 border border-red-600 flex items-center justify-center rounded-sm">
                        <ShieldCheckIcon className="w-6 h-6 text-red-500" />
                    </div>
                    <div>
                        <h2 className="text-3xl font-oswald text-white uppercase tracking-widest text-glow-red">High Command</h2>
                        <div className="text-xs font-mono text-zinc-500 uppercase tracking-widest">Surveillance & Logistics</div>
                    </div>
                </div>
                <button onClick={onClose} className="p-2 hover:bg-zinc-800 rounded-full transition-colors text-zinc-500 hover:text-white">
                    <XMarkIcon className="w-8 h-8" />
                </button>
            </div>

            {/* TAB NAVIGATION */}
            <div className="flex items-center gap-4 mb-6 border-b border-zinc-800">
                <button
                    onClick={() => setActiveTab('activity')}
                    className={`pb-2 px-4 text-sm font-bold uppercase tracking-wider transition-all border-b-2 ${activeTab === 'activity' ? 'text-red-500 border-red-500' : 'text-zinc-500 border-transparent hover:text-zinc-300'}`}
                >
                    Agent Activity
                </button>
                <button
                    onClick={() => setActiveTab('commissions')}
                    className={`pb-2 px-4 text-sm font-bold uppercase tracking-wider transition-all border-b-2 flex items-center gap-2 ${activeTab === 'commissions' ? 'text-amber-500 border-amber-500' : 'text-zinc-500 border-transparent hover:text-zinc-300'}`}
                >
                    Commission Requests
                    {payouts.length > 0 && <span className="bg-amber-500 text-black text-[10px] px-1.5 rounded-full">{payouts.length}</span>}
                </button>
                <button
                    onClick={() => setActiveTab('donations')}
                    className={`pb-2 px-4 text-sm font-bold uppercase tracking-wider transition-all border-b-2 flex items-center gap-2 ${activeTab === 'donations' ? 'text-emerald-500 border-emerald-500' : 'text-zinc-500 border-transparent hover:text-zinc-300'}`}
                >
                    Donation Ledger
                </button>
                <button
                    onClick={() => setActiveTab('financials')}
                    className={`pb-2 px-4 text-sm font-bold uppercase tracking-wider transition-all border-b-2 flex items-center gap-2 ${activeTab === 'financials' ? 'text-blue-500 border-blue-500' : 'text-zinc-500 border-transparent hover:text-zinc-300'}`}
                >
                    Transaction History
                </button>
                <button
                    onClick={() => setActiveTab('logs')}
                    className={`pb-2 px-4 text-sm font-bold uppercase tracking-wider transition-all border-b-2 flex items-center gap-2 ${activeTab === 'logs' ? 'text-green-500 border-green-500' : 'text-zinc-500 border-transparent hover:text-zinc-300'}`}
                >
                    <CommandLineIcon className="w-4 h-4" />
                    System Logs
                </button>
            </div>

            {loading ? (
                <div className="flex-1 flex items-center justify-center text-zinc-500 font-mono animate-pulse min-h-[50vh]">
                    ESTABLISHING SECURE LINK...
                </div>
            ) : (
                <>
                    {/* TAB 1: ACTIVITY */}
                    {activeTab === 'activity' && (
                        <div className="flex-1 flex flex-col md:flex-row gap-6 md:overflow-hidden min-h-0 animate-in fade-in slide-in-from-bottom-2 duration-300">
                            {/* ... Content ... */}
                            {/* LEFT PANEL: STATS & USERS */}
                            <div className="flex-1 flex flex-col gap-6 md:overflow-hidden shrink-0">
                                {/* ... Stats Row ... */}
                                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 shrink-0">
                                    <div className="bg-zinc-900 border border-zinc-700 p-4 rounded-sm">
                                        <div className="text-xs text-zinc-500 uppercase font-bold mb-1 flex items-center gap-2">
                                            <UserGroupIcon className="w-4 h-4" /> Total Agents
                                        </div>
                                        <div className="text-2xl font-mono text-white">{stats?.total_users}</div>
                                    </div>
                                    <div className="bg-zinc-900 border border-zinc-700 p-4 rounded-sm">
                                        <div className="text-xs text-zinc-500 uppercase font-bold mb-1 flex items-center gap-2">
                                            <CurrencyDollarIcon className="w-4 h-4" /> Reserves
                                        </div>
                                        <div className="text-2xl font-mono text-amber-500">{stats?.total_coins.toFixed(0)}</div>
                                    </div>

                                    {/* Referral Graph */}
                                    <div className="bg-zinc-900 border border-zinc-700 p-4 rounded-sm col-span-2 flex flex-col justify-between">
                                        <div className="text-xs text-zinc-500 uppercase font-bold mb-1">Recruitment Source</div>
                                        <div className="flex items-center gap-2 h-4 mt-1 w-full bg-zinc-800 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-red-600"
                                                style={{ width: `${(stats?.referral_stats?.referred / (stats?.total_users || 1)) * 100}%` }}
                                            ></div>
                                            <div className="h-full bg-zinc-600 flex-1"></div>
                                        </div>
                                        <div className="flex justify-between text-[10px] font-mono mt-1">
                                            <span className="text-red-400">Referred: {stats?.referral_stats?.referred}</span>
                                            <span className="text-zinc-400">Independent: {stats?.referral_stats?.independent}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* User List */}
                                <div className="h-[400px] md:h-auto md:flex-1 bg-zinc-900/50 border border-zinc-800 rounded-sm overflow-hidden flex flex-col shrink-0">
                                    <div className="p-3 bg-zinc-900 border-b border-zinc-800 text-xs font-bold text-zinc-500 uppercase tracking-wider sticky top-0 flex justify-between">
                                        <span>Agent Registry</span>
                                        <span className="text-[10px] text-zinc-600">HOVER TO INSPECT</span>
                                    </div>
                                    <div className="flex-1 overflow-y-auto custom-scrollbar p-2 grid grid-cols-1 lg:grid-cols-2 gap-4 content-start">
                                        {users.map(u => (
                                            <div
                                                key={u.id}
                                                onClick={() => handleInspectUser(u)}
                                                className={`
                                        flex flex-col p-4 rounded-sm border cursor-pointer transition-all duration-200 relative overflow-hidden group min-h-[140px]
                                        hover:-translate-y-1 hover:shadow-xl hover:border-red-500/50
                                        ${selectedUser?.id === u.id
                                                        ? 'bg-red-950/20 border-red-900/50 text-white shadow-[0_0_15px_rgba(220,38,38,0.2)]'
                                                        : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:bg-zinc-800'
                                                    }
                                      `}
                                            >
                                                <div className="flex justify-between items-start mb-2">
                                                    <div className="flex flex-col mr-2 flex-1 min-w-0">
                                                        <span className="font-oswald uppercase tracking-widest text-lg text-white group-hover:text-red-500 transition-colors break-words leading-tight">{u.username}</span>
                                                        <span className="text-[9px] font-mono text-zinc-600 uppercase mt-1">ID: {u.id.substring(0, 8)}...</span>
                                                    </div>
                                                    <div className="flex flex-col items-end gap-2">
                                                        <div className="flex items-center gap-2">
                                                            <ShieldCheckIcon className={`w-5 h-5 ${u.role === 'admin' ? 'text-red-500' : 'text-zinc-800'}`} />
                                                            {/* DELETE BUTTON */}
                                                            {u.role !== 'admin' && (
                                                                <button
                                                                    onClick={(e) => handleDeleteUser(e, u.id)}
                                                                    className="text-zinc-700 hover:text-red-600 transition-colors p-1"
                                                                    title="Liquidate Agent"
                                                                >
                                                                    <XMarkIcon className="w-4 h-4" />
                                                                </button>
                                                            )}
                                                        </div>

                                                        {/* Referral Counters */}
                                                        <div className="mt-1 flex flex-col items-end text-[9px] font-mono">
                                                            <span className="text-zinc-500">Ref: <span className="text-white">{u.referrals_count || 0}</span></span>
                                                            <span className="text-zinc-500">Paid: <span className="text-amber-500">{u.paid_referrals_count || 0}</span></span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="flex items-center justify-between mt-auto pt-2 border-t border-white/5">
                                                    <span className={`text-[9px] px-2 py-0.5 rounded border uppercase font-bold tracking-wider
                                                ${u.subscription === 'commander' ? 'text-amber-500 border-amber-900/30 bg-amber-950/10' :
                                                            u.subscription === 'infantry' ? 'text-red-500 border-red-900/30 bg-red-950/10' :
                                                                'text-zinc-500 border-zinc-700 bg-zinc-950'}
                                             `}>
                                                        {u.subscription}
                                                    </span>
                                                    <div className="flex items-center gap-1 text-[9px] text-zinc-500 font-mono">
                                                        <CurrencyDollarIcon className="w-3 h-3 text-amber-900" />
                                                        {u.coins?.toFixed(0) || 0}
                                                    </div>
                                                </div>

                                                {/* Active Indicator Strip */}
                                                {selectedUser?.id === u.id && (
                                                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-red-600"></div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* RIGHT PANEL: CHAT INSPECTOR */}
                            <div className="h-[600px] md:h-auto md:flex-1 bg-black border border-zinc-800 rounded-sm overflow-hidden flex flex-col relative shrink-0">
                                {!selectedUser ? (
                                    <div className="absolute inset-0 flex items-center justify-center text-zinc-700 font-mono text-sm uppercase tracking-widest pointer-events-none">
                                        Select an Agent to Inspect
                                    </div>
                                ) : (
                                    <>
                                        <div className="p-3 bg-zinc-900 border-b border-zinc-800 flex justify-between items-center shrink-0">
                                            <div className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
                                                Transcripts: <span className="text-white">{selectedUser.username}</span>
                                            </div>
                                            <div className="text-[10px] text-zinc-600 font-mono">
                                                {userChats.length} Sessions Found
                                            </div>
                                        </div>
                                        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
                                            {userChats.length === 0 && (
                                                <div className="text-center text-zinc-600 py-10 text-xs font-mono">NO RECORDS FOUND</div>
                                            )}
                                            {userChats.map(session => (
                                                <div key={session.id} className="border border-zinc-800 bg-zinc-950 p-4 rounded-sm">
                                                    <div className="flex justify-between items-center mb-4 border-b border-zinc-900 pb-2">
                                                        <span className="text-xs text-red-500 font-bold uppercase">{session.title}</span>
                                                        <span className="text-[10px] text-zinc-600">{new Date(session.timestamp).toLocaleString()}</span>
                                                    </div>
                                                    <div className="space-y-3">
                                                        {session.messages.map((msg, idx) => (
                                                            <div key={idx} className={`flex ${msg.role === 'model' ? 'justify-end' : 'justify-start'}`}>
                                                                <div className={`max-w-[80%] text-xs font-mono p-2 rounded-sm flex flex-col gap-1 ${msg.role === 'model' ? 'bg-zinc-900 text-zinc-400 border border-zinc-800' : 'bg-zinc-800/50 text-zinc-300'}`}>
                                                                    <div>{msg.parts[0].text}</div>

                                                                    {/* Audio Playback */}
                                                                    {msg.audioUrl && (
                                                                        <div className="mt-2">
                                                                            <audio
                                                                                controls
                                                                                src={msg.audioUrl}
                                                                                className="w-full h-8 opacity-70 hover:opacity-100 transition-opacity"
                                                                            />
                                                                        </div>
                                                                    )}

                                                                    {/* Feedback Display */}
                                                                    {msg.role === 'model' && (msg.feedback || msg.feedbackText) && (
                                                                        <div className="mt-3 pt-2 border-t border-zinc-800 flex flex-col gap-2">
                                                                            <div className="flex items-center justify-between">
                                                                                <div className="flex items-center gap-2">
                                                                                    {msg.feedback === 'like' && <div className="flex items-center gap-1 text-emerald-500 font-bold uppercase tracking-wider text-[10px]"><HandThumbUpIcon className="w-4 h-4" /> Commended</div>}
                                                                                    {msg.feedback === 'dislike' && <div className="flex items-center gap-1 text-red-500 font-bold uppercase tracking-wider text-[10px]"><HandThumbDownIcon className="w-4 h-4" /> Reported</div>}
                                                                                </div>
                                                                            </div>

                                                                            {/* Prominent Comment Box */}
                                                                            {msg.feedbackText && (
                                                                                <div className="bg-zinc-800 border-l-2 border-white pl-3 py-2 pr-2 rounded-r text-zinc-200 text-xs font-sans leading-relaxed shadow-sm">
                                                                                    <span className="text-[9px] text-zinc-500 uppercase font-bold block mb-1">Attached Comment:</span>
                                                                                    "{msg.feedbackText}"
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    )}

                    {/* TAB 5: LOGS */}
                    {activeTab === 'logs' && (
                        <div className="flex-1 flex flex-col gap-6 md:overflow-hidden min-h-0 animate-in fade-in slide-in-from-right-2 duration-300">
                            <div className="bg-[#0c0c0c] border border-zinc-800 rounded-sm overflow-hidden flex flex-col flex-1 shadow-2xl font-mono">

                                {/* Controls */}
                                <div className="p-3 bg-zinc-900/80 border-b border-zinc-800 flex gap-4 items-center">
                                    <div className="text-xs font-bold text-green-500 uppercase tracking-widest flex items-center gap-2">
                                        <CommandLineIcon className="w-4 h-4" />
                                        LIVE TERMINAL
                                    </div>
                                    <div className="h-4 w-[1px] bg-zinc-700"></div>
                                    <div className="flex gap-2">
                                        {(['gpu', 'llm', 'tunnel'] as const).map(src => (
                                            <button
                                                key={src}
                                                onClick={() => setLogSource(src)}
                                                className={`text-[10px] uppercase font-bold px-3 py-1 rounded transition-colors
                                                    ${logSource === src ? 'bg-zinc-700 text-white' : 'bg-black/40 text-zinc-500 hover:text-zinc-300'}
                                                `}
                                            >
                                                {src === 'gpu' ? 'GPU NODE' : src === 'llm' ? 'LLM SERVER' : 'TUNNEL'}
                                            </button>
                                        ))}
                                    </div>
                                    {isLogStreaming && <span className="ml-auto text-[10px] text-green-500 animate-pulse">● STREAMING</span>}
                                </div>

                                {/* Terminal Window */}
                                <div className="flex-1 p-4 overflow-y-auto custom-scrollbar bg-black text-xs space-y-1">
                                    {logs.map((line, idx) => (
                                        <div key={idx} className="whitespace-pre-wrap break-all text-zinc-400 font-mono hover:bg-white/5 hover:text-zinc-200 transition-colors">
                                            <span className="text-zinc-700 select-none mr-2">{(idx + 1).toString().padStart(3, '0')}</span>
                                            {line}
                                        </div>
                                    ))}
                                    <div ref={el => el?.scrollIntoView({ behavior: 'smooth' })}></div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* TAB 2: COMMISSIONS */}
                    {activeTab === 'commissions' && (
                        <div className="flex-1 flex flex-col gap-6 md:overflow-hidden min-h-0 animate-in fade-in slide-in-from-right-2 duration-300">
                            <div className="bg-zinc-900/50 border border-amber-900/50 rounded-sm overflow-hidden flex flex-col flex-1 shadow-2xl">
                                <div className="p-4 bg-amber-950/20 border-b border-amber-900/30 text-sm font-bold text-amber-500 uppercase tracking-wider flex justify-between items-center">
                                    <div className="flex items-center gap-3">
                                        <CurrencyDollarIcon className="w-5 h-5" />
                                        Pending Commission Payment Requests
                                    </div>
                                    <span className="bg-amber-500 text-black px-2 py-0.5 rounded text-xs font-mono">{payouts.length} PENDING</span>
                                </div>
                                <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-3">
                                    {payouts.length === 0 ? (
                                        <div className="flex-1 flex items-center justify-center text-zinc-600 font-mono uppercase tracking-widest text-sm h-full">
                                            No Pending Requisitions
                                        </div>
                                    ) : (
                                        payouts.map(p => (
                                            <div key={p.id} className="bg-zinc-900 border border-zinc-800 p-6 flex items-center justify-between rounded-sm shadow-md group hover:border-amber-900/50 transition-colors">
                                                <div className="flex items-center gap-6">
                                                    <div className="w-12 h-12 bg-amber-950/30 flex items-center justify-center rounded border border-amber-900/30 text-amber-500 font-oswald text-xl">
                                                        $
                                                    </div>
                                                    <div>
                                                        <div className="text-white font-oswald tracking-wide text-xl flex items-center gap-3">
                                                            {p.username}
                                                            <span className="text-amber-500 font-mono text-lg bg-amber-950/30 px-2 py-0.5 rounded border border-amber-900/30">${p.amount.toFixed(2)}</span>
                                                        </div>
                                                        <div className="text-xs text-zinc-500 font-mono mt-1 flex items-center gap-4">
                                                            <span className="flex items-center gap-1"><span className="text-zinc-600">WALLET:</span> <span className="text-zinc-300 font-bold select-all">{p.wallet}</span></span>
                                                            <span className="text-zinc-700">|</span>
                                                            <span className="text-zinc-600">REQ: {new Date(p.timestamp).toLocaleDateString()}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <button
                                                    onClick={() => handleConfirmPayout(p.id)}
                                                    className="bg-amber-600 hover:bg-amber-500 text-black text-sm font-bold uppercase py-3 px-6 rounded-sm transition-transform hover:scale-105 shadow-lg flex items-center gap-2"
                                                >
                                                    <CurrencyDollarIcon className="w-5 h-5" />
                                                    Process Payment
                                                </button>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* TAB 3: DONATIONS */}
                    {activeTab === 'donations' && (
                        <div className="flex-1 flex flex-col gap-6 md:overflow-hidden min-h-0 animate-in fade-in slide-in-from-right-2 duration-300">
                            <div className="bg-zinc-900/50 border border-zinc-800 rounded-sm overflow-hidden flex flex-col flex-1 shadow-2xl">
                                <div className="p-4 bg-zinc-900 border-b border-zinc-800 text-sm font-bold text-emerald-500 uppercase tracking-wider flex justify-between items-center">
                                    <div className="flex items-center gap-3">
                                        <CurrencyDollarIcon className="w-5 h-5" />
                                        Donation History
                                    </div>
                                    <span className="text-zinc-500 text-xs font-mono">TOTAL RECORDS: {donations.length}</span>
                                </div>
                                <div className="flex-1 overflow-y-auto custom-scrollbar">
                                    <table className="w-full text-left border-collapse">
                                        <thead className="bg-zinc-900 sticky top-0 z-10 text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                                            <tr>
                                                <th className="p-4 border-b border-zinc-800">Date</th>
                                                <th className="p-4 border-b border-zinc-800">Agent</th>
                                                <th className="p-4 border-b border-zinc-800 text-right">Amount (USD)</th>
                                                <th className="p-4 border-b border-zinc-800 text-right">Coins Issued</th>
                                                <th className="p-4 border-b border-zinc-800 text-right">Invoice ID</th>
                                            </tr>
                                        </thead>
                                        <tbody className="text-xs font-mono text-zinc-400">
                                            {donations.length === 0 ? (
                                                <tr>
                                                    <td colSpan={5} className="p-8 text-center text-zinc-600 italic">No donations recorded yet.</td>
                                                </tr>
                                            ) : (
                                                donations.map((d, i) => (
                                                    <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-900/50 transition-colors">
                                                        <td className="p-4 text-zinc-500">{new Date(d.date).toLocaleString()}</td>
                                                        <td className="p-4 font-bold text-white">{d.username}</td>
                                                        <td className="p-4 text-right text-emerald-500 font-bold">${(d.amount || 0).toFixed(2)}</td>
                                                        <td className="p-4 text-right text-amber-500 font-bold">{d.coins}</td>
                                                        <td className="p-4 text-right text-zinc-600 font-mono text-[10px]">{d.id}</td>
                                                    </tr>
                                                ))
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* TAB 4: FINANCIALS (TRANSACTIONS) */}
                    {activeTab === 'financials' && (
                        <div className="flex-1 flex flex-col gap-6 md:overflow-hidden min-h-0 animate-in fade-in slide-in-from-right-2 duration-300">
                            <div className="bg-zinc-900/50 border border-zinc-800 rounded-sm overflow-hidden flex flex-col flex-1 shadow-2xl">
                                <div className="p-4 bg-zinc-900 border-b border-zinc-800 text-sm font-bold text-blue-500 uppercase tracking-wider flex justify-between items-center">
                                    <div className="flex items-center gap-3">
                                        <CurrencyDollarIcon className="w-5 h-5" />
                                        Financial Ledger (In/Out)
                                    </div>
                                    <span className="text-zinc-500 text-xs font-mono">TOTAL RECORDS: {transactions.length}</span>
                                </div>
                                <div className="flex-1 overflow-y-auto custom-scrollbar">
                                    <table className="w-full text-left border-collapse">
                                        <thead className="bg-zinc-900 sticky top-0 z-10 text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                                            <tr>
                                                <th className="p-4 border-b border-zinc-800">Date Logged</th>
                                                <th className="p-4 border-b border-zinc-800">Transaction Label</th>
                                                <th className="p-4 border-b border-zinc-800 text-center">Type</th>
                                                <th className="p-4 border-b border-zinc-800 text-right">Amount (USD)</th>
                                                <th className="p-4 border-b border-zinc-800 text-center">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody className="text-xs font-mono text-zinc-400">
                                            {transactions.length === 0 ? (
                                                <tr>
                                                    <td colSpan={5} className="p-8 text-center text-zinc-600 italic">No transactions recorded.</td>
                                                </tr>
                                            ) : (
                                                transactions.map((t: any, i: number) => (
                                                    <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-900/50 transition-colors">
                                                        <td className="p-4 text-zinc-500">{new Date(t.date).toLocaleString()}</td>
                                                        <td className="p-4 font-bold text-white">{t.label}</td>
                                                        <td className="p-4 text-center">
                                                            <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase ${t.type === 'credit' ? 'bg-emerald-950/30 text-emerald-500 border border-emerald-900/50' : 'bg-red-950/30 text-red-500 border border-red-900/50'
                                                                }`}>
                                                                {t.type === 'credit' ? 'INCOME' : 'EXPENSE'}
                                                            </span>
                                                        </td>
                                                        <td className={`p-4 text-right font-bold ${t.type === 'credit' ? 'text-emerald-400' : 'text-red-400'}`}>
                                                            {t.type === 'credit' ? '+' : '-'}${Number(t.amount).toFixed(2)}
                                                        </td>
                                                        <td className="p-4 text-center">
                                                            <span className="text-zinc-500 uppercase text-[9px] border border-zinc-800 px-1 rounded">{t.status}</span>
                                                        </td>
                                                    </tr>
                                                ))
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div >
    );
};

export default AdminDashboard;
