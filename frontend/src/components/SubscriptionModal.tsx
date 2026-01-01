import React from 'react';
import {
    ExclamationTriangleIcon,
    XMarkIcon,
    CheckIcon,
    XCircleIcon,
} from '@heroicons/react/24/solid';
import KampfCoin3D from './KampfCoin3D';
import { User } from '../services/db';

interface SubscriptionModalProps {
    isOpen: boolean;
    onClose: () => void;
    user: User | null;
    onSubscribe: (plan: string) => void;
}

const SubscriptionModal: React.FC<SubscriptionModalProps> = ({ isOpen, onClose, user, onSubscribe }) => {
    if (!isOpen) return null;

    // Local State for Confirmation Step
    const [confirmingPlan, setConfirmingPlan] = React.useState<string | null>(null);
    const [processing, setProcessing] = React.useState(false);

    const handleClick = (plan: string) => {
        setConfirmingPlan(plan);
    };

    const handleConfirm = async () => {
        if (!confirmingPlan) return;
        setProcessing(true);
        try {
            await onSubscribe(confirmingPlan);
            onClose();
        } catch (e) {
            console.error(e);
            alert("Transaction Failed");
        } finally {
            setProcessing(false);
            setConfirmingPlan(null);
        }
    };

    const handleCancel = () => {
        setConfirmingPlan(null);
    };

    if (confirmingPlan) {
        const isUpgrade = user?.subscription !== 'free';
        const price = confirmingPlan === 'infantry' ? '$4.99' : '$9.99';
        const coins = confirmingPlan === 'infantry' ? '100' : '300';

        return (
            <div className="fixed inset-0 z-[160] bg-black/95 backdrop-blur-md flex items-center justify-center p-4">
                <div className="w-full max-w-lg bg-[#0a0a0a] border border-red-500 shadow-[0_0_50px_rgba(220,38,38,0.4)] rounded-lg p-8 text-center animate-in zoom-in duration-300">
                    <h2 className="text-2xl font-oswald text-white uppercase tracking-widest mb-4">
                        Confirm Requisition
                    </h2>
                    <div className="text-zinc-400 font-mono text-sm mb-8 space-y-4">
                        {isUpgrade ? (
                            <p>Proceeding will update your clearance to <span className="text-red-500 font-bold uppercase">{confirmingPlan}</span> and add the <span className="text-amber-500 font-bold">{coins} KC</span> credit pack to your existing balance.</p>
                        ) : (
                            <p>Do you want to buy the <span className="text-red-500 font-bold uppercase">{confirmingPlan}</span> clearance for <span className="text-white font-bold">{price}</span>?</p>
                        )}
                        <p className="text-xs text-zinc-600">Secure transmission line active. 10% Affiliate commission will be processed. Valid for 30 Days.</p>
                    </div>

                    <div className="flex gap-4 justify-center">
                        <button
                            onClick={handleCancel}
                            disabled={processing}
                            className="px-6 py-2 border border-zinc-700 hover:bg-zinc-800 text-zinc-400 font-bold uppercase text-sm tracking-wider rounded-sm transition-colors"
                        >
                            Cancel Operation
                        </button>
                        <button
                            onClick={handleConfirm}
                            disabled={processing}
                            className="px-6 py-2 bg-red-700 hover:bg-red-600 text-white font-bold uppercase text-sm tracking-wider rounded-sm transition-colors border border-red-500 flex items-center gap-2"
                        >
                            {processing ? 'Processing...' : 'Proceed'}
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-[150] bg-black/95 backdrop-blur-md flex items-center justify-center p-4">
            <div className="w-full max-w-6xl bg-[#0a0a0a] border border-red-900/50 shadow-[0_0_50px_rgba(220,38,38,0.2)] rounded-lg relative overflow-hidden flex flex-col max-h-[90vh]">
                {/* Header */}
                <div className="bg-gradient-to-r from-red-950 to-black p-6 border-b border-red-900/30 flex justify-between items-center shrink-0">
                    <div>
                        <h2 className="text-2xl font-oswald text-white uppercase tracking-widest flex items-center gap-2">
                            <ExclamationTriangleIcon className="w-6 h-6 text-red-500 animate-pulse" />
                            Munitions Depleted
                        </h2>
                        <p className="text-red-400 font-mono text-sm mt-1 uppercase">Insufficient Kampf Coins for transmission.</p>
                    </div>
                    <button onClick={onClose} className="text-zinc-500 hover:text-white transition-colors"><XMarkIcon className="w-8 h-8" /></button>
                </div>

                {/* Content */}
                <div className="p-4 md:p-8 grid grid-cols-1 md:grid-cols-4 gap-4 md:gap-6 overflow-y-auto custom-scrollbar">

                    {/* LEFT COLUMN: 3D ANIMATION */}
                    <div className="flex flex-col bg-black rounded-lg border border-white/10 overflow-hidden h-[400px] md:h-full shrink-0">
                        {/* Header */}
                        <div className="p-4 border-b border-white/5 bg-zinc-900/30 shrink-0">
                            <h3 className="text-xl font-oswald text-amber-500 uppercase tracking-widest text-center drop-shadow-lg">
                                Official Currency
                            </h3>
                        </div>

                        {/* Video Body */}
                        <div className="relative flex-1 w-full bg-black/50 min-h-0">
                            <KampfCoin3D className="w-full h-full object-cover" size={undefined} />
                            {/* Subtle inner shadow */}
                            <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_20px_rgba(0,0,0,0.8)]"></div>
                        </div>

                        {/* Footer Description */}
                        <div className="p-4 bg-zinc-900/30 border-t border-white/5 shrink-0">
                            <p className="text-[10px] text-zinc-400 text-center font-mono leading-relaxed">
                                Minted by the High Command.<br />
                                Required for all secure transmissions.
                            </p>
                        </div>
                    </div>

                    {/* Free Tier */}
                    <div className={`border p-6 rounded-lg bg-zinc-900/50 flex flex-col items-center text-center ${user?.subscription === 'free' ? 'border-green-500/50 ring-1 ring-green-500/20' : 'border-zinc-800 opacity-60'}`}>
                        <div className="text-zinc-500 font-bold uppercase mb-2">Conscript</div>
                        <div className="text-3xl font-oswald text-white mb-4">Free</div>
                        <ul className="text-sm text-zinc-400 space-y-2 mb-6 font-mono text-left w-full pl-4">
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-zinc-500" /> 1.0 Coins / Day (~5 Chats)</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-zinc-500" /> Limited Dictator Access</li>
                            <li className="flex items-center gap-2"><XCircleIcon className="w-4 h-4 text-red-900/40" /> No Contextual Memory</li>
                            <li className="flex items-center gap-2"><XCircleIcon className="w-4 h-4 text-red-900/40" /> No Voice Response</li>
                            <li className="flex items-center gap-2"><XCircleIcon className="w-4 h-4 text-red-900/40" /> No Avatar Response</li>
                        </ul>
                        {user?.subscription === 'free' && (
                            <div className="mt-auto text-xs text-green-500 font-bold uppercase tracking-widest flex items-center gap-2">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                Current Status
                            </div>
                        )}
                    </div>

                    {/* Infantry Tier */}
                    <div
                        className={`border  p-6 rounded-lg flex flex-col items-center text-center relative group transition-all ${user?.subscription === 'infantry' ? 'border-green-500/50 bg-green-950/10 ring-1 ring-green-500/20' : 'border-red-900/50 bg-red-950/10 hover:bg-red-950/20 cursor-pointer'}`}
                        onClick={() => user?.subscription !== 'infantry' && handleClick('infantry')}
                    >
                        {!user?.subscription && user?.subscription !== 'infantry' && <div className="absolute top-0 right-0 bg-red-900 text-white text-[10px] px-2 py-0.5 uppercase font-bold">Popular</div>}
                        <div className={`font-bold uppercase mb-2 ${user?.subscription === 'infantry' ? 'text-green-500' : 'text-red-500 group-hover:text-red-400'}`}>Infantry</div>
                        <div className="text-3xl font-oswald text-white mb-4">$4.99<span className="text-sm text-zinc-500">/mo</span></div>
                        <ul className="text-sm text-zinc-300 space-y-2 mb-6 font-mono text-left w-full pl-4">
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-red-500" /> 100 KC (~500 Chats)</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-red-500" /> Access To All Dictator</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-red-500" /> Contextual Conversation</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-red-500" /> Voice Responses</li>
                            <li className="flex items-center gap-2"><XCircleIcon className="w-4 h-4 text-red-900/40" /> No Avatar Response</li>
                            <li className="flex items-center gap-2"><XCircleIcon className="w-4 h-4 text-red-900/40" /> No early Access(New Models)</li>

                        </ul>
                        {user?.subscription === 'infantry' ? (
                            <div className="mt-auto text-xs text-green-500 font-bold uppercase tracking-widest flex items-center gap-2">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                Current Status
                            </div>
                        ) : (
                            <button className="mt-auto w-full py-2 bg-red-700 hover:bg-red-600 text-white font-bold uppercase text-sm tracking-wider rounded-sm transition-colors border border-red-500">
                                Enlist Now
                            </button>
                        )}
                    </div>

                    {/* Commander Tier */}
                    <div
                        className={`border p-6 rounded-lg flex flex-col items-center text-center transition-all ${user?.subscription === 'commander' ? 'border-green-500/50 bg-green-950/10 ring-1 ring-green-500/20' : 'border-amber-600/30 bg-amber-950/10 hover:bg-amber-950/20 cursor-pointer group'}`}
                        onClick={() => user?.subscription !== 'commander' && handleClick('commander')}
                    >
                        <div className={`font-bold uppercase mb-2 ${user?.subscription === 'commander' ? 'text-green-500' : 'text-amber-500 group-hover:text-amber-400'}`}>Commander</div>
                        <div className="text-3xl font-oswald text-white mb-4">$9.99<span className="text-sm text-zinc-500">/mo</span></div>
                        <ul className="text-sm text-zinc-300 space-y-2 mb-6 font-mono text-left w-full pl-4">
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-amber-500" /> 300 KC (~1500 Chats)</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-amber-500" /> Early Access (New Models)</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-amber-500" /> Contextual Conversation</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-amber-500" /> Voice Responses</li>
                            <li className="flex items-center gap-2"><CheckIcon className="w-4 h-4 text-amber-500" /> Avatar Response</li>
                        </ul>
                        {user?.subscription === 'commander' ? (
                            <div className="mt-auto text-xs text-green-500 font-bold uppercase tracking-widest flex items-center gap-2">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                Current Status
                            </div>
                        ) : (
                            <button className="mt-auto w-full py-2 bg-amber-700 hover:bg-amber-600 text-white font-bold uppercase text-sm tracking-wider rounded-sm transition-colors border border-amber-500">
                                Take Command
                            </button>
                        )}
                    </div>

                </div>
                <div className="p-4 bg-black/50 border-t border-zinc-900 text-center text-xs text-zinc-600 font-mono uppercase">
                    Secure transmission encrypton active. By enlisting you agree to the War Room protocols.
                </div>
            </div>
        </div>
    );
};

export default SubscriptionModal;
