import React, { useEffect, useRef, useState } from 'react';
import { XMarkIcon, ArrowDownTrayIcon, ShareIcon, ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/solid';
import html2canvas from 'html2canvas';
import logo from '../assets/logo.jpg';
import propagandaBg from '../assets/propaganda_bg.png';
import classifiedBg from '../assets/classified_bg.png';
import telegramBg from '../assets/telegram_bg.png';
import broadcastBg from '../assets/broadcast_bg.png';
import blueprintBg from '../assets/blueprint_bg.png';

interface ShareData {
    userText: string;
    aiText: string;
    leaderName: string;
    leaderAvatar: string;
    style: string;
    audioUrl?: string; // Optional, might be generated later
}

interface SocialShareModalProps {
    isOpen: boolean;
    onClose: () => void;
    data: ShareData;
    userTier: string;
}

// --- Constants ---
const THEMES = [
    { id: 'classified', name: 'Secret Dossier', desc: 'Confidential File 1945' },
    { id: 'propaganda', name: 'Propaganda', desc: 'State Media Poster' },
    { id: 'telegram', name: 'Telegram', desc: 'Urgent Wire' },
    { id: 'broadcast', name: 'Broadcast', desc: 'Emergency Signal' },
    { id: 'blueprint', name: 'Blueprint', desc: 'Technical Schematic' },
];

const chatBgPattern = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312809/chat_background_u3nybm.webp";

// SVG Data URI for Noise Texture (White Specks) - KEEPING AS OPTIONAL OVERLAY IF NEEDED, BUT MOSTLY BAKED IN NOW
const NOISE_SVG = `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.5' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.5' fill='white'/%3E%3C/svg%3E")`;

const SocialShareModal: React.FC<SocialShareModalProps> = ({ isOpen, onClose, data, userTier }) => {
    const cardRef = useRef<HTMLDivElement>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null); // Not really used for video path but good for fallback
    const [themeIndex, setThemeIndex] = useState(0);
    const [generating, setGenerating] = useState(false);
    const [videoBlob, setVideoBlob] = useState<Blob | null>(null);

    const activeTheme = THEMES[themeIndex];

    // Reset when data changes or modal opens
    useEffect(() => {
        if (isOpen) {
            setThemeIndex(0);
            setVideoBlob(null);
            setPreviewUrl(null);
        }
    }, [isOpen, data.id]);

    // --- Actions ---
    const getContainerBgColor = (themeId: string) => {
        switch (themeId) {
            case 'classified': return '#f0e6d2';
            case 'propaganda': return '#1a1a1a';
            case 'telegram': return '#e8dcb5';
            case 'broadcast': return '#0f172a';
            case 'blueprint': return '#002b59';
            default: return '#000000';
        }
    };

    const handleNextTheme = () => setThemeIndex((prev) => (prev + 1) % THEMES.length);
    const handlePrevTheme = () => setThemeIndex((prev) => (prev - 1 + THEMES.length) % THEMES.length);

    // --- Actions ---
    const captureImage = async () => {
        if (!cardRef.current) return null;
        try {
            const canvas = await html2canvas(cardRef.current, {
                scale: 2, // High Res
                useCORS: true,
                backgroundColor: getContainerBgColor(activeTheme.id),
                logging: false,
            });
            return canvas;
        } catch (err) {
            console.error("Capture failed", err);
            return null;
        }
    };

    const handleGenerateVideo = async () => {
        setGenerating(true);
        try {
            const canvas = await captureImage();
            if (!canvas) throw new Error("Image capture failed");

            const imageBase64 = canvas.toDataURL('image/png');
            setPreviewUrl(imageBase64);

            // 2. Send to Backend
            const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:5000'}/api/generate-video`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('dictator_token')}`
                },
                body: JSON.stringify({
                    image: imageBase64,
                    audioUrl: data.audioUrl,
                    theme: activeTheme.id
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || 'Video generation failed');
            }

            const blob = await response.blob();
            setVideoBlob(blob);

        } catch (err) {
            console.error(err);
            alert("Failed to generate video. " + (err instanceof Error ? err.message : ""));
        } finally {
            setGenerating(false);
        }
    };

    const handleDownloadVideo = () => {
        if (!videoBlob) return;
        const url = window.URL.createObjectURL(videoBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dictator_transmission_${activeTheme.id}_${Date.now()}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    const handleShareVideo = async () => {
        if (!videoBlob) return;
        const file = new File([videoBlob], 'transmission.mp4', { type: 'video/mp4' });
        if (navigator.share && navigator.canShare && navigator.canShare({ files: [file] })) {
            await navigator.share({ files: [file], title: 'Dictator Transmission', text: 'Incoming message.' }).catch(console.error);
        } else {
            alert("Sharing not supported. Please Download.");
        }
    };

    const handleDownloadImage = async () => {
        const canvas = await captureImage();
        if (!canvas) return;
        const url = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = url;
        a.download = `dictator_card_${activeTheme.id}_${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    const handleShareImage = async () => {
        const canvas = await captureImage();
        if (!canvas) return;
        canvas.toBlob(async (blob) => {
            if (!blob) return;
            const file = new File([blob], 'transmission.png', { type: 'image/png' });
            if (navigator.share && navigator.canShare && navigator.canShare({ files: [file] })) {
                await navigator.share({ files: [file], title: 'Dictator Card', text: 'Classified.' }).catch(console.error);
            } else {
                alert("Sharing not supported. Please Download.");
            }
        });
    };

    // --- Theme Styles Mapping ---
    // UPDATED FOR VINTAGE 1945 AESTHETIC
    const getThemeClasses = () => {
        switch (activeTheme.id) {
            case 'classified': return {
                // MANILA FOLDER / DOSSIER STYLE
                bg: 'bg-[#fdf6e3]', // Very light cream/manila
                text: 'text-zinc-900',
                accent: 'text-red-800',
                border: 'border-zinc-900',
                font: 'font-typewriter',
                overlay: '', // No longer needed
                logo: 'contrast-[1.2] sepia-[.3] border-4 border-double border-zinc-800 rotate-[-2deg]',
                msgBg: 'bg-transparent border-l-4 border-zinc-900 pl-4 text-zinc-900',
                userBg: 'bg-zinc-200/50 border border-dashed border-zinc-400 text-zinc-700 italic',
                container: 'bg-[#f0e6d2] border-[12px] border-[#e6dcc0] shadow-[inset_0_0_100px_rgba(0,0,0,0.2)]'
            };
            case 'propaganda': return {
                // CONSTRUCTIVIST POSTER STYLE (Red/Black/White)
                bg: 'bg-[#1a1a1a]', // Dark Grey/Black Base
                text: 'text-[#e6e6e6]',
                accent: 'text-[#d93025]', // Strong Red
                border: 'border-[#e6e6e6]',
                font: 'font-oswald uppercase tracking-widest', // Needs Oswald font
                overlay: '', // No longer needed
                logo: 'contrast-[1.5] border-4 border-white rotate-[2deg]',
                msgBg: 'bg-[#d93025] text-white -skew-x-6 shadow-[10px_10px_0px_rgba(0,0,0,0.5)] border-none p-6',
                userBg: 'bg-zinc-800 text-zinc-400 border-l-4 border-[#d93025]',
                container: 'bg-[#1a1a1a] border-[20px] border-[#d93025]'
            };
            case 'telegram': return {
                // OLD TELEGRAM STYLE
                bg: 'bg-[#e8dcb5]', // Yellowed paper
                text: 'text-[#4a4036]', // Faded ink
                accent: 'text-[#2a2016]',
                border: 'border-[#4a4036]',
                font: 'font-mono uppercase tracking-[0.1em]',
                overlay: '', // No longer needed
                logo: 'sepia-[.8] grayscale contrast-[1.1] border-2 border-[#4a4036]',
                msgBg: 'bg-transparent border-y-2 border-dashed border-[#4a4036]/30 py-4 text-[#2a2016] font-bold',
                userBg: 'bg-transparent text-[#4a4036]/70 text-sm',
                container: 'bg-[#e8dcb5]'
            };
            case 'broadcast': return {
                // CRT / TV SIGNAL
                bg: 'bg-[#0f172a]',
                text: 'text-blue-100',
                accent: 'text-yellow-400',
                border: 'border-blue-500/50',
                font: 'font-sans tracking-wide',
                overlay: '', // No longer needed
                logo: 'grayscale contrast-125 sepia-[.3] border-blue-400',
                msgBg: 'bg-blue-900/40 border border-blue-400/30 text-blue-50 backdrop-blur-sm',
                userBg: 'bg-slate-800/40 text-slate-400 border-l-2 border-slate-600',
                container: 'bg-[#0f172a] shadow-[inset_0_0_50px_rgba(0,0,0,0.8)]'
            };
            case 'blueprint': return {
                // BLUEPRINT
                bg: 'bg-[#002b59]', // Classic Blueprint Blue
                text: 'text-white',
                accent: 'text-white',
                border: 'border-white/70',
                font: 'font-mono',
                overlay: '', // No longer needed
                logo: 'grayscale invert border-2 border-white',
                msgBg: 'bg-[#001f3f] border border-white text-white font-bold',
                userBg: 'bg-[#001f3f]/50 border-l border-white/50 text-white/70',
                container: 'bg-[#002b59]'
            };
            default: return {
                bg: 'bg-zinc-950',
                text: 'text-zinc-400',
                accent: 'text-red-700',
                border: 'border-red-900/30',
                font: 'font-sans',
                overlay: '',
                logo: 'grayscale',
                msgBg: 'bg-black',
                userBg: 'bg-black',
                container: 'bg-black'
            };
        }
    };

    const styles = getThemeClasses();

    // --- RENDER HELPER FOR CARD CONTENT ---
    // We render this twice: once for the visible preview (scaled), and once hidden (fixed size) for high-res capture.
    const renderCard = () => (
        <div
            className={`relative w-[540px] aspect-[4/5] flex flex-col justify-between overflow-hidden shadow-2xl ${styles.container} ${styles.font}`}
            style={{ backgroundColor: getContainerBgColor(activeTheme.id) }} // explicit default color
        >
            {/* --- BACKGROUND LAYERS (FLATTENED IMAGES) --- */}

            {/* CLASSIFIED */}
            {activeTheme.id === 'classified' && (
                <div
                    className="absolute inset-0 bg-cover bg-center z-0"
                    style={{ backgroundImage: `url(${classifiedBg})`, opacity: 0.9 }}
                />
            )}

            {/* TELEGRAM */}
            {activeTheme.id === 'telegram' && (
                <div
                    className="absolute inset-0 bg-cover bg-center z-0"
                    style={{ backgroundImage: `url(${telegramBg})`, opacity: 0.9 }}
                />
            )}

            {/* PROPAGANDA */}
            {activeTheme.id === 'propaganda' && (
                <div
                    className="absolute inset-0 bg-cover bg-center z-0"
                    style={{ backgroundImage: `url(${propagandaBg})`, opacity: 0.9 }}
                />
            )}

            {/* BROADCAST */}
            {activeTheme.id === 'broadcast' && (
                <>
                    <div
                        className="absolute inset-0 bg-cover bg-center z-0"
                        style={{ backgroundImage: `url(${broadcastBg})`, opacity: 0.9 }}
                    />
                    {/* Extra Vignette for Broadcast to enhance glowing feel */}
                    <div className="absolute inset-0 bg-[radial-gradient(circle,transparent_40%,black_100%)] z-0 opacity-60 pointer-events-none"></div>
                </>
            )}

            {/* BLUEPRINT */}
            {activeTheme.id === 'blueprint' && (
                <div
                    className="absolute inset-0 bg-cover bg-center z-0"
                    style={{ backgroundImage: `url(${blueprintBg})`, opacity: 0.9 }}
                />
            )}

            {/* --- FOREGROUND CONTENT LAYERS --- */}

            {/* PRESERVE THE REST OF THE CONTENT HERE (Just ensuring background is set) */}
            {/* Some theme specific overlays like 'Ai' badge for propaganda are baked/or need to be re-added if not baked */}

            {activeTheme.id === 'propaganda' && (
                <div className="absolute top-0 right-0 p-4">
                    <div className="w-16 h-16 border-[4px] border-white rounded-full flex items-center justify-center text-white font-bold text-2xl rotate-12 opacity-80 z-20 shadow-lg">
                        Ai
                    </div>
                </div>
            )}

            {/* 2. MAIN CONTENT */}
            <div className="relative z-10 flex flex-col h-full p-10 justify-between">

                {/* HEADER */}
                <div className={`flex items-start justify-between border-b-2 pb-6 ${activeTheme.id === 'propaganda' ? 'border-white' : 'border-current'} border-opacity-30`}>
                    <div className="flex items-center gap-6">
                        <img
                            src={logo}
                            className={`w-24 h-24 object-cover shadow-xl ${styles.logo}`}
                            alt="Official Seal"
                        />
                        <div>
                            <h1 className={`text-4xl font-bold uppercase leading-[0.85] tracking-tighter ${styles.accent}`}>
                                DICTATORS<br /><span className={styles.text}>.AI</span>
                            </h1>
                            <div className="text-[10px] tracking-[0.4em] uppercase opacity-70 mt-2 font-bold break-words w-40">
                                {activeTheme.id === 'classified' ? 'Official Record' : 'State Transmission'}
                            </div>
                        </div>
                    </div>
                    <div className="text-right opacity-80 pt-2 hidden sm:block">
                        <div className="text-sm font-bold uppercase tracking-widest">Case: #{Math.floor(Math.random() * 9000) + 1000}</div>
                        <div className="text-xs mt-1">{new Date().toLocaleDateString('en-GB').toUpperCase()}</div>
                        <div className="text-xs mt-1">LOC: BUNKER_01</div>
                    </div>
                </div>

                {/* BODY: CONVERSATION */}
                <div className="flex-1 flex flex-col justify-center py-10 gap-10">

                    {/* USER INPUT */}
                    <div className="w-full">
                        <div className={`flex items-center gap-3 mb-2 opacity-60 text-xs font-bold uppercase tracking-widest ${styles.text}`}>
                            <div className="w-2 h-2 bg-current rounded-full"></div>
                            INQUIRY / SUBJECT
                        </div>
                        <div className={`p-6 relative text-2xl leading-tight ${styles.userBg}`}>
                            {/* Decorative Corner for User Input */}
                            {activeTheme.id === 'classified' && <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-zinc-900"></div>}
                            "{data.userText}"
                        </div>
                    </div>

                    {/* AI RESPONSE - The Hero Content */}
                    <div className="w-full flex-1 flex flex-col justify-center">
                        <div className={`flex items-center justify-end gap-3 mb-2 opacity-60 text-xs font-bold uppercase tracking-widest ${styles.accent}`}>
                            DICTATOR RESPONSE
                            <div className={`w-2 h-2 rounded-full ${activeTheme.id === 'propaganda' ? 'bg-white' : 'bg-red-700'}`}></div>
                        </div>
                        {/* Dynamic Text Size Container */}
                        <div className={`p-8 relative w-full ${styles.msgBg} flex items-center justify-center min-h-[160px]`}>
                            {/* Quote Marks */}
                            <div className="absolute -top-4 -left-2 text-6xl opacity-20 font-serif leading-none">“</div>

                            <span className={`relative z-10 leading-tight font-bold uppercase drop-shadow-sm text-center
                                ${data.aiText.length < 50 ? 'text-4xl' :
                                    data.aiText.length < 100 ? 'text-3xl' :
                                        data.aiText.length < 180 ? 'text-2xl' :
                                            data.aiText.length < 250 ? 'text-xl' : 'text-lg'}
                            `}>
                                {data.aiText.substring(0, 300)}
                            </span>

                            <div className="absolute -bottom-8 -right-2 text-6xl opacity-20 font-serif leading-none rotate-180">“</div>
                        </div>
                    </div>

                </div>

                {/* FOOTER */}
                <div className={`pt-4 border-t-2 ${activeTheme.id === 'propaganda' ? 'border-white' : 'border-current'} border-opacity-30 flex justify-between items-center opacity-80`}>
                    <div className="text-[9px] uppercase font-bold max-w-[80%] leading-tight text-justify">
                        Content generated by DICTATORS.AI. It does not spread or support any ideology. For entertainment purpose only.
                    </div>
                    <div className="flex gap-1">
                        {[...Array(3)].map((_, i) => (
                            <div key={i} className={`w-1 h-3 ${activeTheme.id === 'propaganda' ? 'bg-white' : 'bg-zinc-800'}`}></div>
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[60] flex items-start md:items-center justify-center bg-black/95 backdrop-blur-xl animate-in fade-in duration-300 overflow-y-auto">
            {/* Main Scrollable Wrapper */}
            <div className="w-full min-h-full flex items-center justify-center p-0 md:p-4 py-0 md:py-10">

                {/* HIDDEN CAPTURE ELEMENT (Fixed 540px, off-screen) */}
                <div className="fixed top-0 left-0 z-[-50] opacity-0 pointer-events-none">
                    <div ref={cardRef}>
                        {renderCard()}
                    </div>
                </div>

                {/* VISIBLE MODAL CONTENT */}
                <div className="w-full max-w-6xl bg-zinc-900 md:rounded-2xl overflow-hidden shadow-2xl flex flex-col md:flex-row shadow-black/50 border border-white/10 md:border-0 relative">

                    {/* 1. VISIBLE PREVIEW (Top on Mobile, Right on Desktop) */}
                    {/* HIDE on mobile if video is ready, to show only the video/controls and save space */}
                    <div className={`w-full md:flex-1 bg-black relative flex items-start justify-center overflow-hidden order-1 md:order-2 ${videoBlob ? 'hidden md:flex' : 'flex'}`}>
                        {/* Background Pattern for Modal Area */}
                        <div className="absolute inset-0 opacity-20 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] animate-[pulse_10s_ease-in-out_infinite]"></div>
                        <div className="absolute inset-0 bg-gradient-to-b from-black/0 via-black/0 to-black/80 pointer-events-none"></div>

                        {/* THE VISIBLE CARD (Scaled) */}
                        {/* We use origin-top and fixed heights to collapse the whitespace gap caused by scale() */}
                        <div className="relative w-full flex justify-center pt-20 md:pt-12 h-[520px] xs:h-[550px] sm:h-[600px] md:h-auto md:min-h-[600px] transition-all duration-300">
                            <div className="scale-[0.50] xs:scale-[0.60] sm:scale-[0.75] md:scale-90 lg:scale-100 transition-transform duration-300 origin-top">
                                {renderCard()}
                            </div>
                        </div>
                    </div>

                    {/* 2. CONTROLS (Bottom on Mobile, Left on Desktop) */}
                    <div className="w-full md:w-1/3 bg-zinc-950 p-6 md:p-8 pb-24 md:pb-8 flex flex-col gap-6 border-t md:border-t-0 md:border-r border-white/10 shrink-0 order-2 md:order-1 relative z-20">
                        {/* ... (Controls logic remains same) ... */}
                        <div>
                            <h2 className="text-xl font-bold text-white mb-1 flex items-center gap-3">
                                <span className="w-2 h-2 rounded-full bg-red-600 animate-pulse"></span>
                                Secure Channel
                            </h2>
                            <p className="text-zinc-500 text-xs">Configure transmission parameters.</p>
                        </div>
                        {/* Theme Selector - "Immediate Below" */}
                        <div className="bg-zinc-900/50 p-4 rounded-lg border border-white/5">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 block mb-3 text-center">Encryption Protocol (Theme)</label>

                            <div className="flex items-center justify-between gap-2">
                                <button
                                    onClick={handlePrevTheme}
                                    className="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-full transition-colors flex items-center justify-center active:scale-95 bg-black/20"
                                >
                                    <ChevronLeftIcon className="w-5 h-5" />
                                </button>

                                <div className="flex-1 text-center">
                                    <div className="text-white font-bold uppercase tracking-wider text-base animate-in fade-in slide-in-from-right-2 duration-300 key={activeTheme.id}">
                                        {activeTheme.name}
                                    </div>
                                    <div className="text-[9px] text-zinc-500 uppercase tracking-widest mt-1">
                                        {activeTheme.desc}
                                    </div>
                                </div>

                                <button
                                    onClick={handleNextTheme}
                                    className="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-full transition-colors flex items-center justify-center active:scale-95 bg-black/20"
                                >
                                    <ChevronRightIcon className="w-5 h-5" />
                                </button>
                            </div>
                            <div className="flex justify-center gap-1 mt-3">
                                {THEMES.map((_, idx) => (
                                    <div
                                        key={idx}
                                        className={`w-1 h-1 rounded-full transition-all ${idx === themeIndex ? 'bg-white w-3' : 'bg-zinc-700'}`}
                                    />
                                ))}
                            </div>
                        </div>
                        {/* Actions */}
                        <div className="space-y-3 pt-2">

                            {/* --- FREE TIER: IMAGE ONLY --- */}
                            {userTier === 'free' && (
                                <div className="space-y-3">
                                    <div className="p-3 bg-zinc-900 border border-zinc-800 rounded text-center">
                                        <div className="text-[10px] text-zinc-500 font-mono uppercase tracking-widest mb-1">Clearance Level: Conscript</div>
                                        <div className="text-xs text-zinc-400">Image transmission authorized. Video capabilities restricted.</div>
                                    </div>

                                    <button onClick={handleDownloadImage} className="w-full py-4 bg-zinc-100 hover:bg-white text-black font-bold uppercase tracking-widest rounded shadow px-4 flex items-center justify-center gap-3 transition-colors text-sm">
                                        <ArrowDownTrayIcon className="w-5 h-5" /> Download Image
                                    </button>
                                    <button onClick={handleShareImage} className="w-full py-3 bg-zinc-800 hover:bg-zinc-700 text-white font-bold uppercase tracking-widest rounded shadow border border-zinc-700 px-4 flex items-center justify-center gap-2 transition-colors text-xs">
                                        <ShareIcon className="w-4 h-4" /> Share Image
                                    </button>
                                </div>
                            )}

                            {/* --- PAID TIERS: VIDEO FOCUSED --- */}
                            {userTier !== 'free' && (
                                <div className="space-y-3">

                                    {!videoBlob ? (
                                        <>
                                            {/* Primary: Generate Video */}
                                            <button
                                                onClick={handleGenerateVideo}
                                                disabled={generating}
                                                className="w-full py-4 bg-gradient-to-r from-red-900 to-red-800 hover:from-red-800 hover:to-red-700 text-white font-bold uppercase tracking-widest rounded shadow-lg shadow-red-900/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 transition-all active:scale-[0.98]"
                                            >
                                                {generating ? (
                                                    <>
                                                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                                        Encoding...
                                                    </>
                                                ) : (
                                                    <>
                                                        <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                                        Generate Video
                                                    </>
                                                )}
                                            </button>

                                            {/* Secondary: Image Options (keep small/accessible) */}
                                            <div className="grid grid-cols-2 gap-2 opacity-60 hover:opacity-100 transition-opacity">
                                                <button onClick={handleDownloadImage} className="py-2 bg-transparent hover:bg-zinc-800 border border-zinc-800 text-zinc-500 hover:text-white font-bold uppercase rounded text-[10px] tracking-wider flex items-center justify-center gap-2">
                                                    <ArrowDownTrayIcon className="w-3 h-3" /> Img
                                                </button>
                                                <button onClick={handleShareImage} className="py-2 bg-transparent hover:bg-zinc-800 border border-zinc-800 text-zinc-500 hover:text-white font-bold uppercase rounded text-[10px] tracking-wider flex items-center justify-center gap-2">
                                                    <ShareIcon className="w-3 h-3" /> Share
                                                </button>
                                            </div>
                                        </>
                                    ) : (
                                        <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4">
                                            <div className="p-3 bg-green-500/10 border border-green-500/20 rounded text-center">
                                                <div className="text-green-500 text-[10px] font-bold uppercase tracking-widest mb-2">Transmission Ready</div>
                                                <video
                                                    src={window.URL.createObjectURL(videoBlob)}
                                                    className="w-full rounded border border-green-500/20 mb-2 max-h-[40vh] md:max-h-[50vh] object-contain bg-black"
                                                    controls autoPlay loop
                                                    playsInline
                                                />
                                            </div>

                                            {/* Video Actions */}
                                            <div className="grid grid-cols-2 gap-2">
                                                <button onClick={handleDownloadVideo} className="py-3 bg-zinc-100 hover:bg-white text-black font-bold uppercase rounded transition-colors text-xs flex items-center justify-center gap-2">
                                                    <ArrowDownTrayIcon className="w-4 h-4" /> Download Video
                                                </button>
                                                <button onClick={handleShareVideo} className="py-3 bg-zinc-800 hover:bg-zinc-700 text-white font-bold uppercase rounded transition-colors text-xs flex items-center justify-center gap-2">
                                                    <ShareIcon className="w-4 h-4" /> Share Video
                                                </button>
                                            </div>

                                            <button onClick={() => setVideoBlob(null)} className="w-full py-2 text-zinc-500 hover:text-zinc-300 text-[10px] uppercase tracking-wider">
                                                Create New Transmission
                                            </button>
                                        </div>
                                    )}
                                </div>
                            )}

                            <button onClick={onClose} className="w-full py-2 text-zinc-600 hover:text-white font-mono text-[10px] uppercase tracking-widest transition-colors text-center">
                                [ Close Panel ]
                            </button>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};
export default SocialShareModal;
