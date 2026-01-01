import React, { useState, useEffect } from 'react';
import { XMarkIcon, ClipboardDocumentIcon, CheckIcon, CurrencyDollarIcon, LockClosedIcon, LockOpenIcon } from '@heroicons/react/24/solid';
import { User, db } from '../services/db';

interface ReferralModalProps {
    isOpen: boolean;
    onClose: () => void;
    user: User | null;
}

interface CommissionEligibility {
    eligible: boolean;
    days_remaining: number;
    commission_start_date: string | null;
    affiliate_balance: number;
    payout_status: 'pending' | null;
    message: string;
}

const ReferralModal: React.FC<ReferralModalProps> = ({ isOpen, onClose, user }) => {
    const [copied, setCopied] = useState(false);
    const [eligibility, setEligibility] = useState<CommissionEligibility | null>(null);
    const [walletAddress, setWalletAddress] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Fetch eligibility on mount and every 10 seconds
    useEffect(() => {
        if (!isOpen || !user) return;

        const fetchEligibility = async () => {
            try {
                const data = await db.getCommissionEligibility();
                setEligibility(data);
            } catch (e) {
                console.error('Failed to fetch eligibility', e);
            }
        };

        fetchEligibility();
        const interval = setInterval(fetchEligibility, 10000); // Poll every 10 seconds

        return () => clearInterval(interval);
    }, [isOpen, user]);

    if (!isOpen || !user) return null;

    const referralLink = `${window.location.origin}/?ref=${user.username}`;
    const earnings = eligibility?.affiliate_balance ?? user.affiliate_balance ?? 0;
    const isEligible = eligibility?.eligible ?? false;
    const daysRemaining = eligibility?.days_remaining ?? 30;

    const handleCopy = () => {
        navigator.clipboard.writeText(referralLink);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleRedeem = async () => {
        if (!walletAddress.trim()) {
            alert('Please enter a wallet address');
            return;
        }

        setIsSubmitting(true);
        try {
            // Call payout request API (you'll need to add this to db.ts)
            const token = localStorage.getItem('dictator_token');
            const response = await fetch(`${import.meta.env.VITE_API_URL || "VITE_API_URL"}/api/payouts/request`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ wallet: walletAddress })
            });

            const data = await response.json();

            if (response.ok) {
                alert('Payout request submitted successfully! Admin will process it soon.');
                setWalletAddress('');
                onClose();
            } else {
                alert(data.error || 'Failed to submit payout request');
            }
        } catch (e) {
            console.error('Payout request failed', e);
            alert('Failed to submit payout request');
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[150] bg-black/95 backdrop-blur-md flex items-center justify-center p-4">
            <div className="w-full max-w-md bg-[#121212] border border-zinc-700 shadow-2xl relative font-typewriter animate-in fade-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50">
                    <h2 className="text-lg font-oswald uppercase tracking-widest text-emerald-500 flex items-center gap-2">
                        <CurrencyDollarIcon className="w-5 h-5" />
                        Classified Commission
                    </h2>
                    <button onClick={onClose} className="text-zinc-500 hover:text-white transition-colors">
                        <XMarkIcon className="w-6 h-6" />
                    </button>
                </div>

                <div className="p-6 space-y-6">
                    {/* Earnings Display */}
                    <div className="text-center">
                        <div className="text-3xl font-bold text-white mb-1 tracking-tight">${earnings.toFixed(2)}</div>
                        <div className="text-xs text-zinc-500 uppercase tracking-widest">Total Earnings</div>
                    </div>

                    {/* Lock Status */}
                    {eligibility && (
                        <div className={`p-4 border rounded flex items-center gap-3 ${eligibility.payout_status === 'pending'
                            ? 'bg-yellow-900/20 border-yellow-600'
                            : isEligible
                                ? 'bg-emerald-900/20 border-emerald-600'
                                : 'bg-red-900/20 border-red-600'
                            }`}>
                            {eligibility.payout_status === 'pending' ? (
                                <svg className="w-6 h-6 text-yellow-500 flex-shrink-0 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            ) : isEligible ? (
                                <LockOpenIcon className="w-6 h-6 text-emerald-500 flex-shrink-0" />
                            ) : (
                                <LockClosedIcon className="w-6 h-6 text-red-500 flex-shrink-0" />
                            )}
                            <div className="flex-1">
                                <div className={`text-sm font-bold ${eligibility.payout_status === 'pending'
                                    ? 'text-yellow-400'
                                    : isEligible
                                        ? 'text-emerald-400'
                                        : 'text-red-400'
                                    }`}>
                                    {eligibility.payout_status === 'pending'
                                        ? 'Redeem Requested'
                                        : isEligible
                                            ? 'Commission Unlocked'
                                            : `Locked for ${daysRemaining} days`
                                    }
                                </div>
                                <div className="text-xs text-zinc-400 mt-1">{eligibility.message}</div>
                            </div>
                        </div>
                    )}

                    {/* Redeem Section */}
                    {earnings > 0 && eligibility?.payout_status !== 'pending' && (
                        <div className="bg-zinc-900/50 p-4 border border-zinc-800 rounded space-y-3">
                            <label className="text-[10px] uppercase text-zinc-500 font-bold block">Wallet Address (BTC)</label>
                            <input
                                type="text"
                                value={walletAddress}
                                onChange={(e) => setWalletAddress(e.target.value)}
                                placeholder="bc1q..."
                                disabled={!isEligible}
                                className="w-full bg-black p-3 text-sm text-zinc-300 font-mono border border-zinc-700 focus:border-emerald-600 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                            />
                            <button
                                onClick={handleRedeem}
                                disabled={!isEligible || isSubmitting || !walletAddress.trim()}
                                className={`w-full py-3 font-bold uppercase tracking-widest text-sm transition-all ${isEligible && !isSubmitting
                                    ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                                    : 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                                    }`}
                            >
                                {isSubmitting ? 'Submitting...' : isEligible ? 'Redeem Now' : `Redeem in ${daysRemaining} days`}
                            </button>
                        </div>
                    )}

                    {/* Referral Link */}
                    <div className="bg-zinc-900/50 p-4 border border-zinc-800 rounded">
                        <label className="text-[10px] uppercase text-zinc-500 font-bold block mb-2">Your Unique Operations Link</label>
                        <div className="flex items-center gap-2">
                            <code className="flex-1 bg-black p-3 text-xs text-zinc-300 truncate font-mono border border-zinc-700">{referralLink}</code>
                            <button
                                onClick={handleCopy}
                                className={`p-3 border transition-all ${copied ? 'bg-emerald-900/30 border-emerald-600 text-emerald-500' : 'bg-zinc-800 border-zinc-600 hover:bg-zinc-700 text-white'}`}
                                title="Copy Link"
                            >
                                {copied ? <CheckIcon className="w-4 h-4" /> : <ClipboardDocumentIcon className="w-4 h-4" />}
                            </button>
                        </div>
                    </div>

                    {/* Info */}
                    <div className="text-xs text-zinc-400 leading-relaxed text-center border-t border-zinc-800 pt-4">
                        <p>Recruit new agents to the cause.</p>
                        <p className="mt-2 text-emerald-500/80">Earn <span className="font-bold text-emerald-400">10% commission</span> on every subscription paid by operatives you recruit.</p>
                        <p className="mt-2 text-yellow-500/80 text-[10px]">⚠️ Commission unlocks 30 days after first earning.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ReferralModal;
