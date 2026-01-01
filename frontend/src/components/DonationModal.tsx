import React, { useState } from 'react';
import { CurrencyDollarIcon, XMarkIcon } from '@heroicons/react/24/solid';
import { db } from '../services/db';

interface DonationModalProps {
    isOpen: boolean;
    onClose: () => void;
    onPaymentInit: (url: string) => void;
    onError: (msg: string) => void;
}

const DonationModal: React.FC<DonationModalProps> = ({ isOpen, onClose, onPaymentInit, onError }) => {
    const [amount, setAmount] = useState<string>('5');
    const [loading, setLoading] = useState(false);

    const handleDonate = async () => {
        const val = parseFloat(amount);
        if (isNaN(val) || val <= 0) {
            onError("Please enter a valid amount.");
            return;
        }

        setLoading(true);
        try {
            const data = await db.createPayment('donation', val);
            if (data.checkoutLink) {
                onPaymentInit(data.checkoutLink);
            } else {
                throw new Error("Invalid Gateway Response");
            }
        } catch (e: any) {
            console.error("Donation failed", e);
            onError(e.message || "Donation failed. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4 animate-in fade-in duration-200">
            <div className="w-full max-w-md bg-zinc-900 border border-amber-500/30 shadow-[0_0_30px_rgba(245,158,11,0.1)] rounded-lg overflow-hidden relative">

                {/* Header */}
                <div className="p-6 bg-black/40 border-b border-white/5 flex justify-between items-center">
                    <h2 className="text-xl font-oswald text-amber-500 uppercase tracking-widest flex items-center gap-2">
                        <CurrencyDollarIcon className="w-6 h-6" /> Support the Cause
                    </h2>
                    <button onClick={onClose} className="text-zinc-500 hover:text-white transition-colors">
                        <XMarkIcon className="w-6 h-6" />
                    </button>
                </div>

                {/* Body */}
                <div className="p-8 space-y-6">
                    <p className="text-zinc-400 text-sm font-mono leading-relaxed">
                        Your contribution fuels our operations. As a token of gratitude, you will receive <span className="text-amber-500 font-bold">10 KC Coins</span> for every dollar donated.
                    </p>

                    <div className="space-y-2">
                        <label className="text-xs uppercase font-bold text-zinc-500 tracking-wider">Donation Amount (USD)</label>
                        <div className="relative">
                            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-500 font-mono">$</span>
                            <input
                                type="number"
                                min="1"
                                step="1"
                                value={amount}
                                onChange={(e) => setAmount(e.target.value)}
                                className="w-full bg-black/50 border border-zinc-700 rounded p-4 pl-8 text-2xl font-bold text-white focus:outline-none focus:border-amber-500 transition-colors font-mono"
                            />
                        </div>
                    </div>

                    <div className="flex justify-between gap-3 pt-4">
                        <button
                            onClick={onClose}
                            className="flex-1 py-3 border border-zinc-700 text-zinc-400 font-bold uppercase tracking-wider hover:bg-zinc-800 transition-colors rounded hover:text-white"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleDonate}
                            disabled={loading}
                            className="flex-[2] py-3 bg-gradient-to-r from-amber-700 to-amber-600 hover:from-amber-600 hover:to-amber-500 text-white font-bold uppercase tracking-wider transition-all rounded shadow-lg shadow-amber-900/20 disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                    Processing...
                                </>
                            ) : (
                                <>
                                    Confirm Donation
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Footer Decor */}
                <div className="h-1 w-full bg-gradient-to-r from-transparent via-amber-500/50 to-transparent"></div>
            </div>
        </div>
    );
};

export default DonationModal;
