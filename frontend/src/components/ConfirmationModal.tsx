import React from 'react';

interface ConfirmationModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title: string;
    message: React.ReactNode;
}

const ConfirmationModal: React.FC<ConfirmationModalProps> = ({ isOpen, onClose, onConfirm, title, message }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            {/* Modal Content */}
            <div className="relative bg-zinc-900 border-2 border-red-900/50 rounded-sm shadow-[0_0_50px_rgba(0,0,0,0.8)] max-w-md w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200">

                {/* Decorative Top Stripe */}
                <div className="h-1 w-full bg-gradient-to-r from-red-900 via-red-600 to-red-900" />

                <div className="p-6 md:p-8 flex flex-col items-center text-center">
                    {/* Icon / Warning */}
                    <div className="w-12 h-12 rounded-full bg-red-950/30 border border-red-500/30 flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-red-500">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                        </svg>
                    </div>

                    <h3 className="text-2xl font-oswald font-bold text-white uppercase tracking-wider mb-2 text-glow-red">
                        {title}
                    </h3>

                    <div className="text-zinc-400 text-sm font-mono mb-8 leading-relaxed">
                        {message}
                    </div>

                    <div className="flex w-full gap-3">
                        <button
                            onClick={onClose}
                            className="flex-1 py-3 px-4 rounded-sm border border-zinc-700 text-zinc-400 font-mono text-xs font-bold uppercase hover:bg-zinc-800 hover:text-white transition-colors"
                        >
                            Cancel Operation
                        </button>
                        <button
                            onClick={onConfirm}
                            className="flex-1 py-3 px-4 rounded-sm bg-red-900/80 border border-red-700 text-white font-mono text-xs font-bold uppercase hover:bg-red-800 hover:border-red-500 transition-all shadow-[0_0_15px_rgba(220,38,38,0.2)]"
                        >
                            Proceed
                        </button>
                    </div>
                </div>

                {/* Decorative Corner Markers */}
                <div className="absolute top-2 left-2 w-2 h-2 border-t border-l border-red-500/50" />
                <div className="absolute top-2 right-2 w-2 h-2 border-t border-r border-red-500/50" />
                <div className="absolute bottom-2 left-2 w-2 h-2 border-b border-l border-red-500/50" />
                <div className="absolute bottom-2 right-2 w-2 h-2 border-b border-r border-red-500/50" />
            </div>
        </div>
    );
};

export default ConfirmationModal;
