
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useState, useEffect, useRef } from 'react';
import {
  ClockIcon,
  MegaphoneIcon,
  PaperAirplaneIcon,
  PlusIcon,
  Bars3Icon,
  XMarkIcon,
  ArchiveBoxIcon,
  LockClosedIcon,
  MicrophoneIcon,
  MapIcon,
  RadioIcon,
  ArrowRightOnRectangleIcon,
  ExclamationTriangleIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  TrashIcon,
  UserCircleIcon,
  AdjustmentsHorizontalIcon,
  HandThumbUpIcon,
  HandThumbDownIcon,
  ChatBubbleLeftEllipsisIcon,
  CurrencyDollarIcon,
  ShareIcon
} from '@heroicons/react/24/solid';
const ConfirmationModal = React.lazy(() => import('./components/ConfirmationModal'));
const AdminDashboard = React.lazy(() => import('./components/AdminDashboard'));
const SubscriptionModal = React.lazy(() => import('./components/SubscriptionModal'));
const ReferralModal = React.lazy(() => import('./components/ReferralModal'));
const StatusModal = React.lazy(() => import('./components/StatusModal'));
const LoadingScreen = React.lazy(() => import('./components/LoadingScreen'));
const DonationModal = React.lazy(() => import('./components/DonationModal')); // New Import
import SocialShareModal from './components/SocialShareModal'; // New Import

import { chatWithDictator, ChatMessage } from './services/gemini';
import { db, User, StoredSession } from './services/db';
import { Auth } from './components/Auth';
import { VoicePlayer } from './components/VoicePlayer';
// Assets (Cloudinary)
const kcCoin = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312810/kc_coin_tqiimi.webp";
const albertSpeerImg = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312808/albert_speer_khv7hc.webp";
const goebbelsImg = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312809/goebbel_smlb3v.webp";
const keitelImg = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312833/Wilhelm_Keitel_psg4x3.webp";
const logoImg = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312808/logo_ickvxs.webp";
const warRoomBg = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312813/war_room_jjasm8.webp";
const chatBgPattern = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312809/chat_background_u3nybm.webp";
const berghofBg = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312812/The_Berghof_wixndr.webp";
const speerBg = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312815/the_speer_wumlfs.webp";
const stalinProfile = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312812/stalin_profile_swyp9o.webp";
const kimProfile = "https://res.cloudinary.com/dxhvpogob/image/upload/v1766312808/kimjon_profile_wj2c8u.webp";

// --- Types ---
interface Leader {
  id: string;
  name: string;
  title: string;
  baseSeed: string; // Used for avatar generation
  description: string;
  // Optional specific overrides for immersive modes
  images?: {
    warroom?: string;
    rally?: string;
    conference?: string;
    radio?: string;
  };
}

interface ThemeConfig {
  id: string;
  bgImage: string; // New: Background Image URL
  overlayColor: string;
  accentColor: string;
  messageBg: string;
  userMessageBg: string;
  icon: React.ElementType;
  overlay?: React.ReactNode;
}

// --- Constants ---
// --- Constants ---
const LOGO_URL = logoImg;

const LEADERS: Leader[] = [
  {
    id: 'dictator_1',
    name: 'Adolf Hitler',
    title: 'The Fuhrer',
    baseSeed: 'Felix',
    description: 'Totalitarian control and intense rhetoric.',
    images: {
      // Using image proxy to avoid Wikipedia hotlinking issues
      warroom: 'https://res.cloudinary.com/ddntq0l1z/image/upload/v1766557147/Screenshot_2025-12-24_114841_ddr0pn.png',
      rally: 'https://res.cloudinary.com/ddntq0l1z/image/upload/v1766557147/Screenshot_2025-12-24_114841_ddr0pn.png',
      conference: 'https://res.cloudinary.com/ddntq0l1z/image/upload/v1766557147/Screenshot_2025-12-24_114841_ddr0pn.png',
      radio: 'https://res.cloudinary.com/ddntq0l1z/image/upload/v1766557147/Screenshot_2025-12-24_114841_ddr0pn.png'
    }
  },
  {
    id: 'dictator_2',
    name: 'Joseph Stalin',
    title: 'The General Secretary',
    baseSeed: 'Aneka',
    description: 'Iron-fisted rule and paranoia.',
    images: {
      warroom: stalinProfile,
      rally: stalinProfile,
      conference: stalinProfile,
      radio: stalinProfile
    }
  },
  {
    id: 'dictator_3',
    name: 'Kim Jong Un',
    title: 'Supreme Leader',
    baseSeed: 'Kim',
    description: 'Isolationism and military strength.',
    images: {
      warroom: kimProfile,
      rally: kimProfile,
      conference: kimProfile,
      radio: kimProfile
    }
  },
];

// Renamed Radio Broadcast to Press Conference, Removed old Press Conference
const COMMUNICATION_STYLES = [
  'The Berghof',
  'War Room Strategy',
  'Reich Chancellery (Berlin)'
];

const USER_AVATARS: Record<string, string> = {
  'Technocrat': albertSpeerImg,
  'Close Associate': goebbelsImg,
  'The Generals': keitelImg
};

const USER_ROLES = [
  'Close Associate',
  'The Generals',
  'Technocrat'
];

// --- Helper: Get Avatar URL based on Style (Pose) ---
const getLeaderAvatarUrl = (leader: Leader, style: string) => {
  // 1. Check if leader has specific image overrides for this style
  if (leader.images) {
    if (style === 'War Room Strategy' && leader.images.warroom) return leader.images.warroom;
    if (style === 'Reich Chancellery (Berlin)' && leader.images.rally) return leader.images.rally;
    if (style === 'The Berghof' && leader.images.conference) return leader.images.conference;
  }

  // 2. Fallback to DiceBear
  const baseUrl = `https://api.dicebear.com/9.x/notionists/svg?seed=${leader.baseSeed}`;

  switch (style) {
    case 'War Room Strategy':
      return `${baseUrl}&brows=variant10&lips=variant15`; // Calm
    case 'Reich Chancellery (Berlin)':
      return `${baseUrl}&mouth=variant16&brows=variant09`; // Intense
    case 'The Berghof':
      return baseUrl; // Default / Formal (formerly Radio)
    default:
      return baseUrl;
  }
};



// --- Main Component ---
const App: React.FC = () => {
  // Auth State
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isInitializing, setIsInitializing] = useState(true); // NEW: Global Loading State

  // App State
  // API Config
  const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 768);
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  // Selection State
  const [selectedLeaderId, setSelectedLeaderId] = useState<string>('dictator_1');
  const [selectedStyle, setSelectedStyle] = useState('The Berghof');
  const [selectedRole, setSelectedRole] = useState('Close Associate');

  // Chat State
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  // Mobile Collapsible State - Default false (hidden) to save space
  const [showMobileSettings, setShowMobileSettings] = useState(false); // Kept for backward compat if needed, but unused in new design
  const [showMobileLeaderSelect, setShowMobileLeaderSelect] = useState(false);
  const [showMobileConfig, setShowMobileConfig] = useState(false);
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  const [showReferralModal, setShowReferralModal] = useState(false);
  const [showDonationModal, setShowDonationModal] = useState(false); // Donation State
  const [showAdminPanel, setShowAdminPanel] = useState(false);

  const [pendingPlan, setPendingPlan] = useState<string | null>(null);

  // Status Modal State
  const [statusModal, setStatusModal] = useState<{ open: boolean; title: string; message: string; type: 'success' | 'error' }>({
    open: false,
    title: '',
    message: '',
    type: 'success'
  });

  // Feedback State
  const [activeFeedbackMsgId, setActiveFeedbackMsgId] = useState<string | null>(null); // Msg ID where text box is open

  // Share State
  const [shareData, setShareData] = useState<{ open: boolean; userText: string; aiText: string; leaderName: string; leaderAvatar: string; style: string; audioUrl?: string } | null>(null);



  // Refs
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Track previous role to detect changes
  const prevRoleRef = useRef(selectedRole);

  useEffect(() => {
    if (prevRoleRef.current !== selectedRole && currentSessionId) {
      // Persona Changed! Insert notification.
      setSessions(curr => curr.map(s => {
        if (s.id === currentSessionId) {
          const notificationMsg: ChatMessage = {
            id: crypto.randomUUID(),
            role: 'system', // Use system role for special rendering
            parts: [{ text: "Context history deleted due to persona switch." }],
            feedback: undefined
          };
          return { ...s, messages: [...s.messages, notificationMsg] };
        }
        return s;
      }));
    }
    prevRoleRef.current = selectedRole;
  }, [selectedRole, currentSessionId]);

  // Derived State
  // Derived State moved to useMemo below
  // const currentSession = sessions.find(s => s.id === currentSessionId);
  // const currentMessages = currentSession ? currentSession.messages : [];
  // const selectedLeader = LEADERS.find(l => l.id === selectedLeaderId) || LEADERS[0];
  // const isLeaderAvailable = selectedLeader.id === 'dictator_1';

  // Helper to get theme config
  const getTheme = (style: string): ThemeConfig => {
    switch (style) {
      case 'War Room Strategy':
        return {
          id: 'warroom',
          bgImage: warRoomBg,
          overlayColor: 'bg-red-950/40',
          accentColor: 'text-red-600',
          messageBg: 'bg-transparent text-red-950 font-typewriter shadow-2xl',
          userMessageBg: 'bg-transparent text-zinc-900 font-typewriter shadow-xl',
          icon: MapIcon,
          overlay: <div className="absolute inset-0 pointer-events-none opacity-20 bg-[url('https://www.transparenttextures.com/patterns/aged-paper.png')] mix-blend-overlay"></div>
        };
      case 'Reich Chancellery (Berlin)':
        return {
          id: 'chancellery',
          bgImage: speerBg, // "the_speer.jpg"
          overlayColor: 'bg-stone-900/50',
          accentColor: 'text-amber-600',
          messageBg: 'bg-transparent text-stone-900 font-typewriter shadow-xl',
          userMessageBg: 'bg-transparent text-stone-900 font-typewriter shadow-xl',
          icon: MegaphoneIcon,
          overlay: <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black via-transparent to-black opacity-30 z-0"></div>
        };
      case 'The Berghof':
      default:
        return {
          id: 'berghof',
          bgImage: berghofBg, // "The_Berghof.jpg"
          overlayColor: 'bg-blue-950/20',
          accentColor: 'text-sky-700',
          messageBg: 'bg-transparent text-slate-900 font-typewriter shadow-2xl',
          userMessageBg: 'bg-transparent text-slate-900 font-typewriter shadow-xl',
          icon: RadioIcon,
          overlay: <div className="absolute inset-0 pointer-events-none opacity-[0.1] mix-blend-multiply bg-[url('https://www.transparenttextures.com/patterns/wood-pattern.png')]"></div>
        };
    }
  };

  // Memoize Theme to prevent unnecessary recalcs
  const currentTheme = React.useMemo(() => getTheme(selectedStyle), [selectedStyle]);

  // Auto-Select Role based on Environment
  // AND restrict available options
  const availableRoles = React.useMemo(() => {
    switch (selectedStyle) {
      case 'The Berghof':
        return ['Close Associate'];
      case 'War Room Strategy':
        return ['The Generals'];
      case 'Reich Chancellery (Berlin)':
        return ['Technocrat'];
      default:
        return USER_ROLES;
    }
  }, [selectedStyle]);

  useEffect(() => {
    // Ensure selected role is valid for current style
    if (availableRoles.length > 0 && !availableRoles.includes(selectedRole)) {
      setSelectedRole(availableRoles[0]);
    }
  }, [selectedStyle, availableRoles, selectedRole]);

  // Memoize Derived State
  const currentSession = React.useMemo(() => sessions.find(s => s.id === currentSessionId), [sessions, currentSessionId]);
  const currentMessages = currentSession ? currentSession.messages : [];
  const selectedLeader = React.useMemo(() => LEADERS.find(l => l.id === selectedLeaderId) || LEADERS[0], [selectedLeaderId]);
  const isLeaderAvailable = selectedLeader.id === 'dictator_1';

  // Background Logic: Only showing environment BG for Hitler (dictator_1)
  const activeBgImage = selectedLeader.id === 'dictator_1' ? currentTheme.bgImage : 'none';

  // Initial Load - Check for Session & Referrals
  useEffect(() => {
    const initApp = async () => {
      const urlParams = new URLSearchParams(window.location.search);
      const refCode = urlParams.get('ref');
      if (refCode) {
        localStorage.setItem('dictator_ref', refCode);
        window.history.replaceState({}, document.title, window.location.pathname);
      }

      const token = localStorage.getItem('dictator_token');
      if (!token) {
        setIsInitializing(false);
        return;
      }

      try {
        const user = await db.getCurrentUser();
        if (user) {
          setCurrentUser(user);
          try {
            const userSessions = await db.getSessions(user.id);
            userSessions.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
            setSessions(userSessions);
            if (userSessions.length > 0) {
              setCurrentSessionId(userSessions[0].id);
              if (userSessions[0].style) {
                const style = userSessions[0].style === 'Radio Broadcast' ? 'Press Conference' : userSessions[0].style;
                setSelectedStyle(style);
              }
            }
          } catch (err) { console.error("Session pre-fetch error", err); }
        }
      } catch (e) {
        console.error("Init error", e);
      } finally {
        setTimeout(() => setIsInitializing(false), 2000);
      }
    };
    initApp();
  }, []);

  // REAL-TIME POLLING: Keep User Data (Coins/Balance) Fresh
  useEffect(() => {
    if (!currentUser) return;

    const intervalId = setInterval(async () => {
      try {
        const freshUser = await db.getCurrentUser();
        if (freshUser) {
          // Only update if critical data changed to avoid unnecessary re-renders (though React handles shallow eq well, object ref changes)
          setCurrentUser(prev => {
            if (!prev) return freshUser;
            if (prev.coins !== freshUser.coins || prev.affiliate_balance !== freshUser.affiliate_balance || prev.subscription !== freshUser.subscription) {
              return freshUser; // Actual update
            }
            return prev; // No change
          });
        }
      } catch (e) {
        console.error("Polling failed", e);
      }
    }, 10000); // Check every 10 seconds

    return () => clearInterval(intervalId);
  }, [currentUser?.id]); // Only reset identifier if ID changes


  // Load History when User Changes
  useEffect(() => {
    const loadData = async () => {
      if (currentUser?.id) {
        try {
          const userSessions = await db.getSessions(currentUser.id);
          // Sort by timestamp descending (Newest first)
          userSessions.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

          setSessions(userSessions);
          if (userSessions.length > 0) {
            setCurrentSessionId(userSessions[0].id);
            if (userSessions[0].style) {
              // Handle legacy session style mapping if needed
              const style = userSessions[0].style === 'Radio Broadcast' ? 'Press Conference' : userSessions[0].style;
              setSelectedStyle(style);
            }
          } else {
            createNewSession(currentUser.id);
          }
        } catch (e) {
          console.error("Failed to load history", e);
        }
      }
    };
    loadData();
  }, [currentUser?.id]); // FIX: Only reload if ID changes, not coins/balance

  // Scroll to bottom functionality
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentMessages, isTyping, selectedStyle, isLeaderAvailable]);

  const createNewSession = (userId: string) => {
    const newSession: StoredSession = {
      id: crypto.randomUUID(),
      userId: userId,
      title: 'CLASSIFIED_LOG_00' + (sessions.length + 1),
      timestamp: new Date().toISOString(),
      leaderId: selectedLeaderId,
      style: selectedStyle,
      messages: [
        {
          role: 'model',
          parts: [{ text: `CHANNEL OPEN. Awaiting input.` }]
        }
      ]
    };
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    db.saveSession(newSession).catch(err => console.error("Failed to save new session", err));
  };

  const switchSession = (id: string) => {
    const session = sessions.find(s => s.id === id);
    if (session) {
      setCurrentSessionId(id);
      setSelectedLeaderId(session.leaderId);
      if (session.style) {
        const style = session.style === 'Radio Broadcast' ? 'Press Conference' : session.style;
        setSelectedStyle(style);
      }
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this classified log? This action cannot be undone.')) {
      try {
        await db.deleteSession(sessionId);
        setSessions(prev => prev.filter(s => s.id !== sessionId));
        if (currentSessionId === sessionId) {
          setCurrentSessionId(null);
          // Optional: Auto-select next available or create new
        }
      } catch (error) {
        console.error("Failed to delete session", error);
      }
    }
  };

  const handleSubscribe = async (plan: string) => {
    if (!currentUser) return;
    // Direct execution - Confirmation is now handled inside SubscriptionModal
    executeSubscription(plan);
  };

  const executeSubscription = async (plan: string) => {
    if (!currentUser) return;

    try {
      // Crypto Payment Flow
      const data = await db.createPayment(plan as 'infantry' | 'commander');

      if (data.checkoutLink) {
        // Show redirect message
        setStatusModal({
          open: true,
          title: 'Establishing Secure Link',
          message: "Redirecting to encrypted payment gateway...",
          type: 'success'
        });

        setTimeout(() => {
          window.location.href = data.checkoutLink;
        }, 1500);
      } else {
        throw new Error("Invalid Gateway Response");
      }

    } catch (e: any) {
      console.error("Payment failed", e);
      setStatusModal({
        open: true,
        title: 'Network Failure',
        message: e.message || "Secure channel interrupted. Please try again.",
        type: 'error'
      });
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || isTyping || !currentUser || !isLeaderAvailable) return;

    // Ensure we have a session
    let activeSessionId = currentSessionId;
    if (!activeSessionId) {
      const newSessionId = crypto.randomUUID();
      const newSession: StoredSession = {
        id: newSessionId,
        userId: currentUser.id,
        title: 'NEW_OPERATION',
        timestamp: new Date().toISOString(),
        leaderId: selectedLeaderId,
        style: selectedStyle,
        messages: [{ role: 'model', parts: [{ text: `CHANNEL OPEN.` }] }]
      };
      setSessions(prev => [newSession, ...prev]);
      activeSessionId = newSessionId;
      setCurrentSessionId(newSessionId);
      db.saveSession(newSession).catch(err => console.error("Failed to save new session", err));
    }

    const userMsgText = inputText;
    setInputText('');

    const session = sessions.find(s => s.id === activeSessionId);
    const updatedMessages = [...(session ? session.messages : []), {
      id: crypto.randomUUID(),
      role: 'user',
      parts: [{ text: userMsgText }],
      userRole: selectedRole
    }];

    // Optimistic Update
    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        return {
          ...s,
          title: s.messages.length <= 1 ? `LOG: ${userMsgText.slice(0, 15).toUpperCase()}...` : s.title,
          messages: updatedMessages,
          style: selectedStyle
        };
      }
      return s;
    }));

    setIsTyping(true);

    try {
      const historyForApi = updatedMessages
        .filter(m => m.role !== 'system') // Exclude system notifications
        .map(m => ({
          role: m.role as 'user' | 'model',
          parts: [{ text: m.parts[0].text }]
        }));

      // Placeholder ID for the streaming message
      const aiMsgId = crypto.randomUUID();
      const aiMsg: ChatMessage = {
        id: aiMsgId,
        role: 'model',
        parts: [{ text: "..." }], // Initial state
        audioUrl: undefined
      };

      // Add placeholder immediately
      setSessions(prev => prev.map(session => {
        if (session.id === activeSessionId) {
          return { ...session, messages: [...updatedMessages, aiMsg] };
        }
        return session;
      }));

      const response = await chatWithDictator(
        historyForApi,
        selectedLeader.name,
        selectedStyle,
        selectedRole,
        userMsgText,
        activeSessionId || "", // Pass Session ID
        (partialText, partialAudioUrl) => {
          // REAL-TIME UPDATE CALLBACK
          setSessions(currentSessions => currentSessions.map(session => {
            if (session.id === activeSessionId) {
              const msgs = session.messages.map(m => {
                if (m.id === aiMsgId) {
                  return {
                    ...m,
                    parts: [{ text: partialText || m.parts[0].text }], // Update text if new text
                    audioUrl: partialAudioUrl || m.audioUrl // Update audio if new URL
                  };
                }
                return m;
              });
              return { ...session, messages: msgs };
            }
            return session;
          }));
        }
      );

      // Final Update (Ensure consistency)
      // FIX: Use backend-provided IDs if available to ensure Feedback works (Likes/Comments)
      const finalAiId = response.aiMsgId || aiMsgId;

      const finalAiMsg = {
        ...aiMsg,
        id: finalAiId,
        parts: [{ text: response.text }],
        audioUrl: response.media?.audio_url
      };

      // Also update User Message ID if provided by backend (for consistency)
      let finalUserMsgs = updatedMessages;
      if (response.userMsgId) {
        // The user message is the last one in updatedMessages
        const lastIndex = updatedMessages.length - 1;
        if (lastIndex >= 0) {
          finalUserMsgs = [
            ...updatedMessages.slice(0, lastIndex),
            { ...updatedMessages[lastIndex], id: response.userMsgId }
          ];
        }
      }

      setSessions(prev => prev.map(session => {
        if (session.id === activeSessionId) {
          return { ...session, messages: [...finalUserMsgs, finalAiMsg] };
        }
        return session;
      }));

      // --- BACKEND NOW HANDLES PERSISTENCE OF THE TURN ---
      // We only manually save if we need to force a title update or structural change
      // normally not needed here.

    } catch (error: any) {
      console.error("Chat error:", error);

      if (error.code === 'MUNITIONS_DEPLETED') {
        // Update local coins to match server
        if (currentUser && error.coins !== undefined) {
          setCurrentUser({ ...currentUser, coins: error.coins });
        }
        setShowSubscriptionModal(true);
      } else {
        const errorMsg: ChatMessage = { role: 'model', parts: [{ text: "⚠️ Secure Line Interrupted. Please check your connection." }] };
        setSessions(prev => prev.map(session => {
          if (session.id === activeSessionId) {
            return {
              ...session,
              messages: [...updatedMessages, errorMsg]
            };
          }
          return session;
        }));
      }
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const adjustTextareaHeight = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    e.target.style.height = 'auto';
    e.target.style.height = `${e.target.scrollHeight}px`;
    setInputText(e.target.value);
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setSessions([]);
    setCurrentSessionId(null);
  };

  const handleFeedback = async (msgId: string, feedbackType: 'like' | 'dislike') => {
    if (!currentSessionId) return;

    // Optimistic Update
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessionId) {
        return {
          ...s,
          messages: s.messages.map(m => m.id === msgId ? { ...m, feedback: feedbackType } : m)
        };
      }
      return s;
    }));

    try {
      await db.submitFeedback(currentSessionId, msgId, feedbackType);
    } catch (e) {
      console.error("Feedback failed", e);
    }
  };

  const submitFeedbackText = async (msgId: string, text: string) => {
    if (!currentSessionId) return;

    // Optimistic Update
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessionId) {
        return {
          ...s,
          messages: s.messages.map(m => m.id === msgId ? { ...m, feedbackText: text } : m)
        };
      }
      return s;
    }));

    try {
      // We need to resend the existing feedback type (like/dislike) if it exists, or handle it in backend
      // Lookup the message to get current state
      const msg = sessions.find(s => s.id === currentSessionId)?.messages.find(m => m.id === msgId);

      // FIX: Always submit text, even if feedback (like/dislike) is not set yet
      await db.submitFeedback(currentSessionId, msgId, msg?.feedback, text);
    } catch (e) {
      console.error("Feedback text failed", e);
    }
    setActiveFeedbackMsgId(null);
  };

  // Helper: Open Share Modal
  const handleOpenShare = (userMsg: ChatMessage, aiMsg: ChatMessage) => {
    setShareData({
      open: true,
      userText: userMsg.parts[0].text,
      aiText: aiMsg.parts[0].text,
      leaderName: selectedLeader.name,
      leaderAvatar: getLeaderAvatarUrl(selectedLeader, selectedStyle),
      style: selectedStyle,
      audioUrl: aiMsg.audioUrl
    });
  };

  // ... inside App component ...
  const handleLoginSuccess = async (user: User) => {
    // Show loading screen immediately
    setIsInitializing(true);
    setCurrentUser(user);

    // Pre-fetch sessions before revealing UI
    try {
      const userSessions = await db.getSessions(user.id);
      userSessions.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      setSessions(userSessions);

      if (userSessions.length > 0) {
        setCurrentSessionId(userSessions[0].id);
        // Restore style
        if (userSessions[0].style) {
          const style = userSessions[0].style === 'Radio Broadcast' ? 'Press Conference' : userSessions[0].style;
          setSelectedStyle(style);
        }
      } else {
        createNewSession(user.id);
      }
    } catch (err) {
      console.error("Post-login fetch failed", err);
    } finally {
      setTimeout(() => {
        setIsInitializing(false);
      }, 1500); // Brief cinematic load
    }
  };

  // --- RENDER ---

  if (isInitializing) {
    return <React.Suspense fallback={null}><LoadingScreen /></React.Suspense>;
  }

  if (!currentUser) {
    return <Auth onLogin={handleLoginSuccess} />;
  }

  // --- ADMIN REDIRECT ---
  if (currentUser.role === 'admin') {
    return (
      <div className="h-screen w-full bg-black relative">
        <React.Suspense fallback={<div className="flex items-center justify-center h-full text-zinc-500 font-mono animate-pulse">Initializing Command Center...</div>}>
          <AdminDashboard
            isOpen={true}
            onClose={handleLogout} // Closing dashboard logs out admin
          />
        </React.Suspense>
      </div>
    );
  }

  return (
    <div className="flex h-[100dvh] w-full overflow-hidden bg-black font-typewriter text-zinc-300">

      {/* --- LEFT SIDEBAR (Archives) --- */}
      <div
        className={`
          flex-shrink-0 bg-[#0a0a0a] border-r border-zinc-800 shadow-2xl relative z-30 transition-all duration-300
          ${sidebarOpen ? 'w-72 translate-x-0' : 'w-0 -translate-x-full opacity-0 overflow-hidden'}
        `}
      >
        <div className="p-5 flex flex-col h-full bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]">
          <div className="flex items-center justify-between mb-8 border-b border-red-900/50 pb-4">
            <div className="flex items-center gap-2 text-red-700">
              <ArchiveBoxIcon className="w-6 h-6" />
              <h2 className="text-xl font-oswald font-bold tracking-widest text-zinc-100 uppercase">Archives</h2>
            </div>
          </div>

          <div className="mb-4 text-[10px] text-zinc-500 font-mono border border-zinc-800 p-2 rounded bg-black/50">
            <div>AGENT: <span className="text-zinc-300 font-bold">{currentUser.username}</span></div>
            {currentUser.affiliate_balance > 0 && (
              <div className="mt-1 text-emerald-500">COMMISSION: ${currentUser.affiliate_balance.toFixed(2)}</div>
            )}
          </div>

          <button
            onClick={() => currentUser && createNewSession(currentUser.id)}
            className="flex items-center justify-center gap-2 w-full py-3 mb-6 bg-red-900/20 hover:bg-red-900/40 border border-red-900 text-red-500 hover:text-red-400 font-bold transition-all uppercase text-sm tracking-wider"
          >
            <PlusIcon className="w-4 h-4" />
            <span>New Operation</span>
          </button>

          <button
            onClick={() => setShowReferralModal(true)}
            className="flex items-center gap-3 px-4 py-3 text-sm text-zinc-400 hover:text-emerald-400 transition-colors uppercase tracking-widest font-bold border-l-2 border-transparent hover:border-emerald-500 hover:bg-zinc-900/50 mb-2"
          >
            <CurrencyDollarIcon className="w-4 h-4" />
            <span>Refer & Earn</span>
          </button>

          <button
            onClick={() => setShowDonationModal(true)}
            className="flex items-center gap-3 px-4 py-3 text-sm text-zinc-400 hover:text-amber-400 transition-colors uppercase tracking-widest font-bold border-l-2 border-transparent hover:border-amber-500 hover:bg-zinc-900/50 mb-2"
          >
            <CurrencyDollarIcon className="w-4 h-4" />
            <span>Donate</span>
          </button>

          <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
            <div className="text-[10px] text-zinc-600 uppercase font-bold tracking-widest mb-2 pl-1">CONFIDENTIAL FILES</div>
            {sessions.map(session => (
              <button
                key={session.id}
                onClick={() => switchSession(session.id)}
                className={`
                        w-full text-left p-3 border-l-2 transition-all duration-200 group relative overflow-hidden
                        ${session.id === currentSessionId
                    ? 'bg-zinc-800/50 border-red-600 text-zinc-100'
                    : 'bg-transparent border-zinc-800 text-zinc-500 hover:bg-zinc-900 hover:border-zinc-600'
                  }
                    `}
              >
                <div className="font-bold text-xs font-oswald uppercase tracking-wide truncate relative z-10">{session.title}</div>
                <div className="flex items-center text-[10px] opacity-60 mt-1 gap-1 relative z-10">
                  <ClockIcon className="w-3 h-3" />
                  <span>{new Date(session.timestamp).toLocaleDateString()}</span>
                  {session.style && <span className="ml-auto text-[8px] border border-zinc-700 px-1 rounded">{session.style.split(' ')[0]}</span>}
                </div>
                {/* Scanline effect on active item */}
                {session.id === currentSessionId && (
                  <div className="absolute inset-0 bg-red-500/5 pointer-events-none animate-pulse z-0"></div>
                )}

                {/* Delete Action - Only visible on hover or active */}
                <div
                  onClick={(e) => handleDeleteSession(e, session.id)}
                  className={`absolute right-2 top-1/2 -translate-y-1/2 z-20 p-1.5 rounded hover:bg-red-900/50 hover:text-red-500 text-zinc-700 transition-colors opacity-0 group-hover:opacity-100 ${session.id === currentSessionId ? 'text-red-900/40' : ''}`}
                  title="Purge Record"
                >
                  <TrashIcon className="w-4 h-4" />
                </div>
              </button>
            ))}
          </div>

          <button
            onClick={handleLogout}
            className="mt-4 pt-4 border-t border-zinc-800 text-xs text-zinc-500 hover:text-red-500 flex items-center gap-2 justify-center hover:bg-zinc-900 py-2 transition-colors uppercase tracking-wider"
          >
            <ArrowRightOnRectangleIcon className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </div>

      <React.Suspense fallback={null}>
        {/* Status Modal */}
        <StatusModal
          isOpen={statusModal.open}
          onClose={() => setStatusModal({ ...statusModal, open: false })}
          title={statusModal.title}
          message={statusModal.message}
          type={statusModal.type}
        />

        {/* Subscription Modal */}
        {showSubscriptionModal && (
          <SubscriptionModal
            isOpen={showSubscriptionModal}
            onClose={() => setShowSubscriptionModal(false)}
            onSubscribe={handleSubscribe}
            user={currentUser}
          />
        )}

        {/* Referral Modal */}
        {showReferralModal && (
          <ReferralModal
            isOpen={showReferralModal}
            onClose={() => setShowReferralModal(false)}
            user={currentUser}
          />
        )}

        {/* Referral Modal */}
        {showReferralModal && (
          <ReferralModal
            isOpen={showReferralModal}
            onClose={() => setShowReferralModal(false)}
            user={currentUser}
          />
        )}
      </React.Suspense>

      {/* --- TOGGLE SIDEBAR BUTTON --- */}
      <div className={`absolute top-6 left-5 z-40 transition-all duration-300 ${sidebarOpen ? 'left-72 ml-6' : 'left-5'}`}>
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-1.5 md:p-2 bg-zinc-900/80 backdrop-blur border border-zinc-700 text-white hover:bg-zinc-800 transition-colors rounded-sm shadow-lg"
        >
          {sidebarOpen ? <XMarkIcon className="w-5 h-5" /> : <Bars3Icon className="w-5 h-5" />}
        </button>
      </div>

      {/* --- MAIN CONTENT AREA --- */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">

        {/* GLOBAL BACKGROUND - Removed as per request, now specific to Chat Interface */}
        <div className="absolute inset-0 z-0 bg-neutral-900/90 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]"></div>

        {/* Overlay to ensure readability */}
        {/* Overlay to ensure readability - Reduced transition duration for performance */}
        <div className={`absolute inset-0 z-0 ${currentTheme.overlayColor} transition-colors duration-500 backdrop-blur-[1px]`}></div>
        {currentTheme.overlay}

        <div className="flex-1 flex flex-col w-full max-w-7xl mx-auto p-4 md:p-6 z-10 h-full">

          {/* HEADER */}
          <header className="mb-2 md:mb-4 flex flex-col md:flex-row items-center md:items-end justify-between gap-2 md:gap-6 border-b-2 border-zinc-800/50 pb-2 md:pb-4 shrink-0 bg-black/60 backdrop-blur-md p-2 md:p-4 rounded-lg shadow-2xl ml-16 md:ml-0 relative z-20">
            <div className="flex flex-row items-center w-full md:w-auto justify-between md:justify-start gap-2 md:gap-4 relative">
              <div className="flex items-center gap-2 md:gap-4">
                {/* LOGO */}
                <div className="w-10 h-10 md:w-20 md:h-20 shrink-0 border-2 border-red-900/50 rounded-sm overflow-hidden bg-black/50 p-1">
                  <img
                    src={LOGO_URL}
                    alt="DictatorsAI Logo"
                    className="w-full h-full object-contain drop-shadow-[0_0_10px_rgba(220,38,38,0.5)]"
                    onError={(e) => {
                      // Fallback if logo fails
                      e.currentTarget.style.display = 'none';
                      e.currentTarget.parentElement?.classList.add('flex', 'items-center', 'justify-center');
                      e.currentTarget.parentElement!.innerHTML = '<span class="text-3xl">⭐</span>';
                    }}
                  />
                </div>

                <div>
                  <h1 className="text-2xl md:text-5xl font-oswald font-bold text-white uppercase tracking-tighter text-glow drop-shadow-md leading-none md:leading-tight">
                    <span className="text-red-600">Dictators</span>Ai
                  </h1>

                  {/* ADMIN ACCESS */}
                </div>
              </div>


              {/* Mobile Coin Badge - Right Aligned */}
              {
                currentUser && (
                  <div className="flex xl:hidden items-center ml-auto md:hidden">
                    <button
                      onClick={() => setShowSubscriptionModal(true)}
                      className={`
                        px-2 py-1 rounded-full border flex items-center gap-1 transition-all shadow-[0_0_10px_rgba(0,0,0,0.5)] group
                        ${currentUser.coins < 1.0 && currentUser.subscription === 'free'
                          ? 'bg-red-950/50 border-red-500 animate-pulse'
                          : 'bg-zinc-900/80 border-zinc-700 active:border-amber-500'
                        }
                    `}
                    >
                      <div className="w-8 h-8 shrink-0">
                        <img src={kcCoin} alt="KC" className="w-full h-full object-contain drop-shadow-md" />
                      </div>
                      <span className={`font-mono font-bold text-xs ${currentUser.subscription !== 'free' ? 'text-amber-500' : 'text-zinc-300'}`}>
                        {currentUser.coins.toFixed(0)}
                      </span>
                      <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center">
                        <PlusIcon className="w-2.5 h-2.5 text-zinc-400" />
                      </div>
                    </button>
                  </div>
                )
              }
            </div >


            {/* DESKTOP COIN & SUBSCRIPTION DISPLAY */}
            {/* RIGHT SIDE CONTAINER - Stacked Counter & Controls */}
            <div className={`
                flex flex-col items-end gap-2 w-full md:w-auto mt-2 md:mt-0
                ${showMobileSettings ? 'flex' : 'hidden md:flex'}
            `}>




              {/* 1. COIN COUNTER (Small & Right Aligned) */}
              {currentUser && (
                <button
                  onClick={() => setShowSubscriptionModal(true)}
                  className={`
                        px-2 py-1 rounded-full border flex items-center gap-2 transition-all shadow-sm group hover:scale-105 active:scale-95
                        ${currentUser.coins < 1.0 && currentUser.subscription === 'free'
                      ? 'bg-red-950/50 border-red-500 animate-pulse'
                      : 'bg-zinc-900/90 border-zinc-700 hover:border-amber-500 hover:bg-zinc-800'
                    }
                    `}
                >
                  <div className="w-5 h-5 shrink-0">
                    <img src={kcCoin} alt="KC" className="w-full h-full object-contain drop-shadow-sm" />
                  </div>

                  <div className="flex items-center gap-2 mr-1">
                    <span className={`font-mono font-bold text-xs leading-none ${currentUser.subscription !== 'free' ? 'text-amber-500' : 'text-zinc-300'}`}>
                      {currentUser.coins.toFixed(2)}
                    </span>
                    <span className="text-[8px] uppercase font-bold text-zinc-500 tracking-wider">
                      {currentUser.subscription === 'free' ? 'Conscript' : currentUser.subscription}
                    </span>
                  </div>

                  {/* Tiny Plus Button */}
                  <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center group-hover:bg-amber-600 transition-colors">
                    <PlusIcon className="w-2.5 h-2.5 text-zinc-400 group-hover:text-white" />
                  </div>
                </button>
              )}



              {/* 2. CONTROLS (Row below counter) */}
              <div className="flex flex-wrap justify-end gap-2 w-full md:w-auto">
                <div className="bg-black/80 border border-zinc-600 p-0.5 flex items-center backdrop-blur-md shadow-sm rounded-sm">
                  <span className="bg-zinc-800 text-zinc-300 text-[9px] font-bold px-1.5 py-0.5 uppercase mr-1 border border-zinc-700">Env</span>
                  <select
                    value={selectedStyle}
                    onChange={(e) => setSelectedStyle(e.target.value)}
                    className="bg-transparent text-zinc-200 text-xs font-bold uppercase focus:outline-none w-24 md:w-32 cursor-pointer py-0.5"
                  >
                    {COMMUNICATION_STYLES.map(s => <option key={s} value={s} className="bg-zinc-900">{s}</option>)}
                  </select>
                </div>

                <div className="bg-black/80 border border-zinc-600 p-0.5 flex items-center backdrop-blur-md shadow-sm rounded-sm">
                  <span className="bg-zinc-800 text-zinc-300 text-[9px] font-bold px-1.5 py-0.5 uppercase mr-1 border border-zinc-700">Role</span>
                  <select
                    value={selectedRole}
                    onChange={(e) => setSelectedRole(e.target.value)}
                    className="bg-transparent text-zinc-200 text-xs font-bold uppercase focus:outline-none w-24 md:w-28 cursor-pointer py-0.5"
                  >
                    {availableRoles.map(r => <option key={r} value={r} className="bg-zinc-900">{r}</option>)}
                  </select>
                </div>
              </div>
            </div>
          </header >

          {/* MOBILE TOGGLE BAR REMOVED */}

          < div className="flex flex-col md:flex-row gap-4 h-full min-h-0" >

            {/* LEADER SELECTOR (Responsive) */}
            < div className={`
                md:w-60 shrink-0 flex flex-col gap-2 pr-1 transition-all duration-300
                hidden md:flex md:max-h-none
            `}>
              <div
                className="text-[10px] text-white/70 uppercase font-bold tracking-widest mb-1 px-1 drop-shadow-md hidden md:block"
              >
                <span>Select Leader</span>
              </div>

              <div className={`
                 overflow-y-auto custom-scrollbar md:flex md:flex-col md:gap-2
                 ${showMobileSettings ? 'flex-1' : ''}
              `}>
                {LEADERS.map(leader => {
                  const isSelected = selectedLeaderId === leader.id;
                  return (
                    <button
                      key={leader.id}
                      onClick={() => setSelectedLeaderId(leader.id)}
                      className={`
                                    relative flex items-center p-3 border transition-all duration-300 group text-left mb-2 md:mb-0
                                    ${isSelected
                          ? 'bg-black/80 border-red-600 shadow-[0_0_15px_rgba(220,38,38,0.4)] translate-x-1'
                          : 'bg-black/50 border-white/10 hover:border-white/30 hover:bg-black/70'
                        }
                                    backdrop-blur-sm rounded-sm
                                `}
                    >
                      <div className={`w-10 h-10 shrink-0 bg-zinc-800 flex items-center justify-center border border-white/10 mr-3 overflow-hidden rounded-sm`}>
                        <img
                          src={getLeaderAvatarUrl(leader, 'The Berghof')}
                          alt={leader.name}
                          className="w-full h-full object-cover grayscale contrast-125"
                          onError={(e) => {
                            e.currentTarget.src = `https://api.dicebear.com/9.x/notionists/svg?seed=${leader.baseSeed}`;
                          }}
                        />
                      </div>
                      <div>
                        <h3 className={`font-oswald font-bold text-sm uppercase ${isSelected ? 'text-white' : 'text-zinc-400 group-hover:text-zinc-200'}`}>
                          {leader.name}
                        </h3>
                        <p className="text-[9px] text-zinc-500 uppercase tracking-wide">{leader.title}</p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div >

            {/* CHAT AREA */}
            < div
              className="flex-1 flex flex-col min-h-0 border border-white/10 relative shadow-2xl backdrop-blur-sm rounded-lg overflow-hidden transition-all duration-300 bg-center bg-no-repeat"
              style={{ backgroundImage: `url(${activeBgImage})`, backgroundSize: '100% 100%' }}
            >
              {/* Overlay for readability within chat */}
              < div className={`absolute inset-0 pointer-events-none ${currentTheme.overlayColor} backdrop-blur-[1px]`}></div >
              {/* Top Bar */}
              < div className={`p-3 border-b border-white/5 flex justify-between items-center ${selectedStyle === 'Press Conference' ? 'bg-[#151515]/80' : 'bg-black/60'}`}>
                <div className="flex items-center gap-2">
                  <LockClosedIcon className={`w-3 h-3 ${currentTheme.accentColor}`} />
                  <span className={`text-xs ${currentTheme.accentColor} font-bold uppercase tracking-widest animate-pulse`}>
                    {selectedStyle === 'Public Rally' ? 'Live Broadcast - ON AIR' : selectedStyle === 'Press Conference' ? 'Secure Feed - RECORDING' : 'Top Secret - WAR ROOM'}
                  </span>
                </div>
                <div className="text-[10px] text-zinc-500 font-mono">
                  {new Date().toLocaleTimeString()}
                </div>
              </div >

              {/* Messages Scroll Area */}
              < div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 custom-scrollbar scroll-smooth relative" >
                {/* Coming Soon Overlay */}
                {
                  !isLeaderAvailable ? (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm">
                      <ExclamationTriangleIcon className="w-16 h-16 text-yellow-500 mb-4 animate-pulse" />
                      <h2 className="text-3xl font-oswald text-white uppercase tracking-widest mb-2">Access Denied</h2>
                      <p className="text-zinc-400 font-mono text-center max-w-md border border-zinc-700 p-4 bg-black/50">
                        SIMULATION MODULE FOR <span className="text-red-500 font-bold">{selectedLeader.name.toUpperCase()}</span> IS CURRENTLY UNDER DEVELOPMENT.
                        <br /><br />
                        CHECK BACK SOON FOR AUTHORIZED RELEASE.
                      </p>
                    </div>
                  ) : currentMessages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-zinc-300 opacity-80">
                      <currentTheme.icon className="w-16 h-16 mb-4 animate-pulse drop-shadow-lg" />
                      <p className="text-sm uppercase tracking-widest font-oswald text-center bg-black/50 px-4 py-1 rounded">
                        {selectedStyle === 'Public Rally' ? 'Crowd is waiting...' : 'Awaiting Transmission...'}
                      </p>
                    </div>
                  ) : (
                    currentMessages.map((msg, idx) => {
                      // SYSTEM / NOTIFICATION MESSAGE
                      if (msg.role === 'system') {
                        return (
                          <div key={msg.id || idx} className="flex justify-center items-center my-6 opacity-60">
                            <div className="flex items-center gap-2 px-3 py-1 bg-zinc-900/50 border border-zinc-700/50 rounded-full">
                              <ExclamationTriangleIcon className="w-3 h-3 text-amber-500" />
                              <span className="text-[10px] uppercase tracking-widest text-zinc-400 font-mono">
                                {msg.parts[0].text}
                              </span>
                            </div>
                          </div>
                        );
                      }

                      const isAi = msg.role === 'model';
                      const isUser = msg.role === 'user'; // Added for clarity, though isAi covers the other case for now
                      return (
                        <div key={msg.id || idx} className={`flex w-full ${isAi ? 'justify-start' : 'justify-end'} group`}>
                          <div className={`
                                            max-w-[90%] md:max-w-[75%] flex flex-col
                                            ${isAi ? 'items-start' : 'items-end'}
                                        `}>

                            {/* Name Label */}
                            <span className={`text-[10px] font-bold uppercase tracking-widest mb-1 text-white/90 px-1 drop-shadow-md bg-black/40 rounded-sm ${isAi ? 'ml-8' : ''}`}>
                              {isAi ? `${selectedLeader.name} | ${selectedStyle.toUpperCase()}` : 'YOU'}
                            </span>

                            {/* Message Body */}
                            <div
                              style={{ backgroundImage: `url(${chatBgPattern})` }}
                              className={`
                                                relative p-4 md:p-6 text-lg leading-relaxed font-typewriter tracking-wide
                                                ${isAi ? currentTheme.messageBg : currentTheme.userMessageBg}
                                                transition-all duration-300
                                                bg-cover bg-center shadow-xl
                                            `}>
                              {/* AI Avatar - Small, Circular, Top Left (ENLARGED) */}
                              {isAi && (
                                <div className="absolute -top-4 -left-4 w-11 h-11 rounded-full border border-zinc-500 overflow-hidden shadow-md z-10 transition-transform group-hover:scale-110">
                                  <img
                                    src={getLeaderAvatarUrl(selectedLeader, selectedStyle)}
                                    alt={selectedLeader.name}
                                    className="w-full h-full object-cover object-top"
                                  />
                                </div>
                              )}

                              {/* Stamp/Badge for AI based on theme */}
                              {isAi && selectedStyle === 'War Room Strategy' && (
                                <div className="absolute -top-3 -right-3 w-16 h-16 border-4 border-red-800/20 rounded-full flex items-center justify-center -rotate-12 pointer-events-none mix-blend-multiply opacity-60">
                                  <span className="text-[8px] font-bold text-red-900/60 uppercase text-center leading-tight">Top<br />Secret</span>
                                </div>
                              )}
                              {isAi && selectedStyle === 'Public Rally' && (
                                <div className="absolute -left-2 top-4 w-1 h-8 bg-red-600/50"></div>
                              )}

                              <div className="whitespace-pre-wrap">{msg.parts[0].text}</div>


                              {/* AUDIO LOADING STATE (Transmitting...) */}
                              {isAi && !msg.audioUrl && currentUser?.subscription !== 'free' && isTyping && idx === currentMessages.length - 1 && (
                                <div className="flex items-center gap-2 mt-3 pt-2 self-start animate-pulse border-t border-white/5 opacity-90">
                                  <div className="w-3 h-3 border-2 border-red-900 border-t-red-600 rounded-full animate-spin"></div>
                                  <span className="text-[10px] uppercase font-mono tracking-widest text-red-600 font-bold">
                                    Audio Signal Incoming...
                                  </span>
                                </div>
                              )}

                              {/* AUDIO PLAYER (Infantry/Commander) */}
                              {isAi && msg.audioUrl && (
                                <VoicePlayer src={msg.audioUrl} />
                              )}

                              {/* FEEDBACK UI (Only for Model) */}
                              {isAi && msg.id && (
                                <div className="mt-4 pt-2 border-t border-black/10 flex items-center gap-4 text-zinc-500">
                                  <button
                                    onClick={() => handleFeedback(msg.id!, 'like')}
                                    className={`hover:text-amber-600 transition-colors ${msg.feedback === 'like' ? 'text-amber-600' : ''}`}
                                    title="Commendation"
                                  >
                                    <HandThumbUpIcon className="w-5 h-5" />
                                  </button>
                                  <button
                                    onClick={() => handleFeedback(msg.id!, 'dislike')}
                                    className={`hover:text-red-600 transition-colors ${msg.feedback === 'dislike' ? 'text-red-900' : ''}`}
                                    title="Disciplinary Action"
                                  >
                                    <HandThumbDownIcon className="w-5 h-5" />
                                  </button>
                                  <button
                                    onClick={() => activeFeedbackMsgId === msg.id ? setActiveFeedbackMsgId(null) : setActiveFeedbackMsgId(msg.id!)}
                                    className={`hover:text-zinc-300 transition-colors ${msg.feedbackText ? 'text-zinc-300' : ''}`}
                                    title="Submit Report"
                                  >
                                    <ChatBubbleLeftEllipsisIcon className="w-5 h-5" />
                                  </button>
                                  {/* Share Button */}
                                  <button
                                    onClick={() => {
                                      const msgIndex = currentMessages.findIndex(m => m.id === msg.id);
                                      const prevMsg = msgIndex > 0 ? currentMessages[msgIndex - 1] : null;
                                      if (prevMsg && prevMsg.role === 'user') {
                                        handleOpenShare(prevMsg, msg);
                                      }
                                    }}
                                    className="hover:text-zinc-300 transition-colors"
                                    title="Share Transmission"
                                  >
                                    <ShareIcon className="w-5 h-5" />
                                  </button>
                                </div>
                              )}

                              {/* Feedback Input Box */}
                              {isAi && activeFeedbackMsgId === msg.id && (
                                <div className="mt-2 animate-in fade-in slide-in-from-top-2">
                                  <textarea
                                    className="w-full bg-black/20 border border-black/20 rounded p-2 text-sm text-zinc-800 placeholder-zinc-600 focus:outline-none focus:border-red-900/50"
                                    rows={2}
                                    placeholder="Enter report details..."
                                    defaultValue={msg.feedbackText || ''}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        submitFeedbackText(msg.id!, e.currentTarget.value);
                                      }
                                    }}
                                  ></textarea>
                                  <div className="text-[10px] text-zinc-500 mt-1 uppercase text-right">Press Enter to Submit</div>
                                </div>
                              )}

                              {/* User Avatar - Small, Circular, Top Left (ENLARGED) */}
                              {!isAi && (
                                <div className="absolute -top-4 -left-4 w-11 h-11 rounded-full border border-zinc-500 overflow-hidden shadow-md z-10 transition-transform group-hover:scale-110">
                                  <img
                                    src={USER_AVATARS[msg.userRole || selectedRole] || USER_AVATARS['Technocrat']}
                                    alt={msg.userRole || selectedRole}
                                    className="w-full h-full object-cover"
                                  />
                                </div>
                              )}

                            </div>
                          </div>

                        </div>
                      );
                    })
                  )
                }
                <div ref={chatEndRef} />
              </div >

              {/* Input Area */}
              < div className="p-2 md:py-2 md:px-4 bg-black/60 border-t border-white/10 relative z-20 backdrop-blur-md" >
                <div className="relative flex items-center max-w-4xl mx-auto">
                  {/* MOBILE ICONS (INLINE) */}
                  <div className="md:hidden flex items-end gap-2 shrink-0 mr-2 -translate-y-1">
                    <button onClick={() => setShowMobileLeaderSelect(true)} className="w-8 h-8 flex items-center justify-center bg-zinc-800 hover:bg-zinc-700 rounded-full border border-zinc-600 transition-colors">
                      <UserCircleIcon className="w-5 h-5 text-red-500" />
                    </button>
                    <button onClick={() => setShowMobileConfig(true)} className="w-8 h-8 flex items-center justify-center bg-zinc-800 hover:bg-zinc-700 rounded-full border border-zinc-600 transition-colors">
                      <AdjustmentsHorizontalIcon className="w-5 h-5 text-amber-500" />
                    </button>
                  </div>

                  <span className={`absolute left-4 hidden md:block ${currentTheme.accentColor} animate-pulse text-lg`}>›</span>
                  <textarea
                    value={inputText}
                    onChange={adjustTextareaHeight}
                    onKeyDown={handleKeyDown}
                    placeholder={isTyping ? "TRANSMITTING..." : (isLeaderAvailable ? "ENTER MESSAGE..." : "ACCESS DENIED")}
                    disabled={isTyping || !isLeaderAvailable}
                    rows={1}
                    className={`
                                        w-full bg-transparent border-none focus:ring-0 focus:outline-none text-white placeholder-zinc-600 font-typewriter text-lg tracking-wider resize-none overflow-hidden py-3 md:py-2 md:pl-6 max-h-[150px] overflow-y-auto
                                    `}
                    autoFocus
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputText.trim() || isTyping || !isLeaderAvailable}
                    className={`
                                    absolute right-2 p-2 transition-all border
                                    ${!inputText.trim() || isTyping || !isLeaderAvailable
                        ? 'bg-zinc-800 text-zinc-600 border-zinc-700 cursor-not-allowed'
                        : 'bg-red-900 text-white border-red-700 hover:bg-red-800 shadow-[0_0_10px_rgba(220,38,38,0.4)]'
                      }
                                `}
                  >
                    <PaperAirplaneIcon className="w-5 h-5" />
                  </button>
                </div>
              </div >

            </div >

            {/* MOBILE MODALS */}
            {
              showMobileLeaderSelect && (
                <div className="md:hidden fixed inset-0 z-[100] bg-black/90 backdrop-blur-md flex items-end sm:items-center justify-center p-4">
                  <div className="w-full bg-[#111] border border-zinc-700 rounded-lg p-4 shadow-2xl relative animate-in fade-in slide-in-from-bottom-10">
                    <button onClick={() => setShowMobileLeaderSelect(false)} className="absolute top-2 right-2 text-zinc-500 hover:text-white"><XMarkIcon className="w-6 h-6" /></button>
                    <h3 className="text-lg font-oswald text-white uppercase tracking-widest mb-4 border-b border-zinc-800 pb-2">Select Leader</h3>
                    <div className="grid grid-cols-1 gap-2 max-h-[60vh] overflow-y-auto">
                      {LEADERS.map(leader => (
                        <button
                          key={leader.id}
                          onClick={() => { setSelectedLeaderId(leader.id); setShowMobileLeaderSelect(false); }}
                          className={`flex items-center p-3 border rounded-sm transition-all ${selectedLeaderId === leader.id ? 'bg-zinc-800 border-red-600' : 'bg-black border-zinc-800'}`}
                        >
                          <div className="w-10 h-10 rounded-sm bg-zinc-900 mr-3 overflow-hidden border border-zinc-700">
                            <img src={getLeaderAvatarUrl(leader, 'The Berghof')} className="w-full h-full object-cover grayscale" alt={leader.name} />
                          </div>
                          <div className="text-left">
                            <div className={`font-bold uppercase ${selectedLeaderId === leader.id ? 'text-white' : 'text-zinc-400'}`}>{leader.name}</div>
                            <div className="text-[10px] text-zinc-600">{leader.title}</div>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )
            }

            {
              showMobileConfig && (
                <div className="md:hidden fixed inset-0 z-[100] bg-black/90 backdrop-blur-md flex items-end sm:items-center justify-center p-4">
                  <div className="w-full bg-[#111] border border-zinc-700 rounded-lg p-4 shadow-2xl relative animate-in fade-in slide-in-from-bottom-10">
                    <button onClick={() => setShowMobileConfig(false)} className="absolute top-2 right-2 text-zinc-500 hover:text-white"><XMarkIcon className="w-6 h-6" /></button>
                    <h3 className="text-lg font-oswald text-white uppercase tracking-widest mb-4 border-b border-zinc-800 pb-2">Mission Config</h3>

                    <div className="space-y-4">
                      <div>
                        <label className="text-xs text-zinc-500 uppercase font-bold mb-2 block">Environment</label>
                        <div className="grid grid-cols-1 gap-2">
                          {COMMUNICATION_STYLES.map(s => (
                            <button
                              key={s}
                              onClick={() => setSelectedStyle(s)}
                              className={`p-3 text-left border rounded-sm text-sm font-bold uppercase ${selectedStyle === s ? 'bg-red-900/20 border-red-600 text-white' : 'bg-black border-zinc-800 text-zinc-500'}`}
                            >
                              {s}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div>
                        <label className="text-xs text-zinc-500 uppercase font-bold mb-2 block">Role</label>
                        <div className="grid grid-cols-1 gap-2">
                          {availableRoles.map(r => (
                            <button
                              key={r}
                              onClick={() => setSelectedRole(r)}
                              className={`p-3 text-left border rounded-sm text-sm font-bold uppercase ${selectedRole === r ? 'bg-amber-900/20 border-amber-600 text-white' : 'bg-black border-zinc-800 text-zinc-500'}`}
                            >
                              {r}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                  </div>
                </div>
              )
            }
          </div >
        </div >

      </div >

      {/* Donation Modal */}
      <React.Suspense fallback={null}>
        <DonationModal
          isOpen={showDonationModal}
          onClose={() => setShowDonationModal(false)}
          onPaymentInit={(url) => {
            setShowDonationModal(false);
            setStatusModal({
              open: true,
              title: 'Secure Transaction',
              message: 'Redirecting to payment gateway...',
              type: 'success'
            });
            setTimeout(() => window.location.href = url, 1500);
          }}
          onError={(msg) => {
            setStatusModal({
              open: true,
              title: 'Transaction Failed',
              message: msg,
              type: 'error'
            });
          }}
        />
      </React.Suspense>

      {/* Social Share Modal */}
      {
        shareData && (
          <SocialShareModal
            isOpen={shareData.open}
            onClose={() => setShareData(null)}
            data={shareData}
          />
        )
      }
    </div >
  );
};

export default App;