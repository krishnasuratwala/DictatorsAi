
import React, { useState, useRef, useEffect } from 'react';
import { PlayIcon, PauseIcon, SpeakerWaveIcon } from '@heroicons/react/24/solid';

interface VoicePlayerProps {
    src: string;
    theme?: string; // Optional: To pass 'red', 'amber' etc. based on leader validation if needed later
}

export const VoicePlayer: React.FC<VoicePlayerProps> = React.memo(({ src }) => {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const updateProgress = () => {
            if (audio.duration) {
                setProgress((audio.currentTime / audio.duration) * 100);
            }
        };

        const setAudioData = () => {
            setDuration(audio.duration);
            setIsLoaded(true);
        };

        const handleEnded = () => {
            setIsPlaying(false);
            setProgress(0);
        };

        audio.addEventListener('timeupdate', updateProgress);
        audio.addEventListener('loadedmetadata', setAudioData);
        audio.addEventListener('ended', handleEnded);

        // Reset state if src changes
        setIsPlaying(false);
        setProgress(0);

        return () => {
            audio.removeEventListener('timeupdate', updateProgress);
            audio.removeEventListener('loadedmetadata', setAudioData);
            audio.removeEventListener('ended', handleEnded);
        };
    }, [src]);

    const togglePlay = () => {
        if (!audioRef.current) return;
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            audioRef.current.play();
        }
        setIsPlaying(!isPlaying);
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!audioRef.current) return;
        const seekTime = (parseFloat(e.target.value) / 100) * duration;
        audioRef.current.currentTime = seekTime;
        setProgress(parseFloat(e.target.value));
    };

    const formatTime = (time: number) => {
        if (isNaN(time)) return "0:00";
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    };

    return (
        <div className="w-full max-w-[280px] md:max-w-[320px] bg-black/40 border border-zinc-700/50 rounded p-2 flex items-center gap-3 backdrop-blur-sm mt-3 group hover:border-red-900/50 transition-colors">
            <audio ref={audioRef} src={src} preload="metadata" />

            <button
                onClick={togglePlay}
                className="w-8 h-8 flex items-center justify-center bg-zinc-800 hover:bg-zinc-700 rounded-full border border-zinc-600 transition-colors shrink-0"
            >
                {isPlaying ? (
                    <PauseIcon className="w-4 h-4 text-white" />
                ) : (
                    <PlayIcon className="w-4 h-4 text-white ml-0.5" />
                )}
            </button>

            <div className="flex-1 flex flex-col justify-center gap-1">
                <div className="flex justify-between items-center text-[9px] uppercase font-bold text-zinc-500 tracking-wider">
                    <div className="flex items-center gap-1">
                        <SpeakerWaveIcon className={`w-3 h-3 ${isPlaying ? 'text-red-500 animate-pulse' : 'text-zinc-600'}`} />
                        <span>Voice Log</span>
                    </div>
                    <span>{formatTime(audioRef.current?.currentTime || 0)} / {formatTime(duration)}</span>
                </div>

                {/* Custom Range Slider using Tailwind accent */}
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={progress}
                    onChange={handleSeek}
                    className="
                        w-full h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer
                        [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 
                        [&::-webkit-slider-thumb]:bg-red-600 [&::-webkit-slider-thumb]:rounded-full 
                        [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-125
                    "
                />
            </div>
        </div>
    );
});
