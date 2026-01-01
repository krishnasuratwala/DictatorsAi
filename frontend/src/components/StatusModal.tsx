import React from 'react';
import { CheckBadgeIcon, XCircleIcon } from '@heroicons/react/24/solid';

interface StatusModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    message: string;
    type: 'success' | 'error';
}

const StatusModal: React.FC<StatusModalProps> = ({ isOpen, onClose, title, message, type }) => {
    if (!isOpen) return null;

    const isSuccess = type === 'success';

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            {/* Modal Content */}
            <div className={`relative bg-zinc-900 border-2 ${isSuccess ? 'border-emerald-900/50' : 'border-red-900/50'} rounded-sm shadow-[0_0_50px_rgba(0,0,0,0.8)] max-w-sm w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200`}>

                {/* Decorative Top Stripe */}
                <div className={`h-1 w-full bg-gradient-to-r ${isSuccess ? 'from-emerald-900 via-emerald-500 to-emerald-900' : 'from-red-900 via-red-600 to-red-900'}`} />

                <div className="p-6 md:p-8 flex flex-col items-center text-center">
                    {/* Icon */}
                    <div className={`w-12 h-12 rounded-full mb-4 flex items-center justify-center ${isSuccess ? 'bg-emerald-950/30 border border-emerald-500/30' : 'bg-red-950/30 border border-red-500/30'}`}>
                        {isSuccess ? (
                            <CheckBadgeIcon className="w-6 h-6 text-emerald-500" />
                        ) : (
                            <XCircleIcon className="w-6 h-6 text-red-500" />
                        )}
                    </div>

                    <h3 className={`text-xl font-oswald font-bold uppercase tracking-wider mb-2 ${isSuccess ? 'text-emerald-500' : 'text-red-500'}`}>
                        {title}
                    </h3>

                    <div className="text-zinc-400 text-sm font-mono mb-6 leading-relaxed">
                        {message}
                    </div>

                    <button
                        onClick={onClose}
                        className={`w-full py-2 px-4 rounded-sm font-mono text-xs font-bold uppercase transition-all shadow-lg
                            ${isSuccess
                                ? 'bg-emerald-900/20 hover:bg-emerald-900/40 border border-emerald-700 text-emerald-500 hover:text-emerald-400'
                                : 'bg-red-900/20 hover:bg-red-900/40 border border-red-700 text-red-500 hover:text-red-400'
                            }
                        `}
                    >
                        Acknowledge
                    </button>
                </div>

                {/* Decorative Corner Markers */}
                <div className={`absolute top-2 left-2 w-2 h-2 border-t border-l ${isSuccess ? 'border-emerald-500/30' : 'border-red-500/30'}`} />
                <div className={`absolute top-2 right-2 w-2 h-2 border-t border-r ${isSuccess ? 'border-emerald-500/30' : 'border-red-500/30'}`} />
                <div className={`absolute bottom-2 left-2 w-2 h-2 border-b border-l ${isSuccess ? 'border-emerald-500/30' : 'border-red-500/30'}`} />
                <div className={`absolute bottom-2 right-2 w-2 h-2 border-b border-r ${isSuccess ? 'border-emerald-500/30' : 'border-red-500/30'}`} />
            </div>
        </div>
    );
};

export default StatusModal;
