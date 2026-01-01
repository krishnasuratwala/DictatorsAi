import React, { useEffect, useState } from 'react';

const LOADING_MESSAGES = [
    "ESTABLISHING SECURE CONNECTION...",
    "VERIFYING CLEARANCE CODES...",
    "DECRYPTING ARCHIVES...",
    "SYNCHRONIZING WITH HIGH COMMAND...",
    "LOADING PERSONALITY MATRIX..."
];

const LoadingScreen: React.FC = () => {
    const [text, setText] = useState("");
    const [msgIndex, setMsgIndex] = useState(0);

    useEffect(() => {
        const fullText = LOADING_MESSAGES[msgIndex];
        let charIndex = 0;

        const typeInterval = setInterval(() => {
            if (charIndex <= fullText.length) {
                setText(fullText.slice(0, charIndex));
                charIndex++;
            } else {
                clearInterval(typeInterval);
                setTimeout(() => {
                    setMsgIndex(prev => (prev + 1) % LOADING_MESSAGES.length);
                }, 1000);
            }
        }, 50);

        return () => clearInterval(typeInterval);
    }, [msgIndex]);

    return (
        <div className="fixed inset-0 bg-black z-[200] flex flex-col items-center justify-center font-typewriter text-red-600">
            <div className="w-64 h-2 bg-zinc-900 border border-red-900/50 rounded-full overflow-hidden mb-8 relative">
                <div className="absolute inset-0 bg-red-600/20 animate-pulse"></div>
                <div className="h-full bg-red-600 animate-[loading_2s_ease-in-out_infinite]"></div>
            </div>

            <div className="text-xl tracking-widest font-bold animate-pulse">
                {text}<span className="animate-blink">_</span>
            </div>

            <div className="absolute bottom-10 text-[10px] text-zinc-600 font-mono tracking-[0.2em] uppercase">
                Dictator AI Systems v2.0
            </div>

            <style>{`
                @keyframes loading {
                    0% { width: 0%; transform: translateX(-100%); }
                    50% { width: 100%; transform: translateX(0); }
                    100% { width: 100%; transform: translateX(100%); }
                }
            `}</style>
        </div>
    );
};

export default LoadingScreen;
