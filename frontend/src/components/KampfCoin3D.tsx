import React from 'react';
// import coinVideo from '../assets/kampf_coin.mp4';
const coinVideo = "https://res.cloudinary.com/dxhvpogob/video/upload/v1766312822/kampf_coin_dshnhi.mp4"; // Placeholder - PLEASE REPLACE

interface KampfCoin3DProps {
    size?: number; // Size in pixels
    className?: string;
}

const KampfCoin3D: React.FC<KampfCoin3DProps> = ({ size, className = '' }) => {
    return (
        <div
            className={`relative group ${className} flex items-center justify-center`}
            style={size ? { width: size, height: size } : undefined}
        >
            <video
                src={coinVideo}
                autoPlay
                loop
                muted
                playsInline
                preload="auto"
                className="w-full h-full object-cover shadow-[0_0_20px_rgba(217,119,6,0.4)] transition-opacity duration-500"
                style={{ transform: 'translateZ(0)', willChange: 'transform' }}
            />
        </div>
    );
};

export default KampfCoin3D;
