/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

export interface ChatMessage {
  id?: string;
  role: 'user' | 'model' | 'system';
  parts: { text: string }[];
  userRole?: string;
  feedback?: 'like' | 'dislike';
  feedbackText?: string;
  audioUrl?: string; // URL for generated audio
}

// Configuration for your Custom LLM
// We use a relative path "/chat" so the Vite Proxy (in vite.config.ts) can forward it 
// to http://127.0.0.1:5000/chat, bypassing CORS issues.
const DEFAULT_LLM_URL = "/chat";
const CUSTOM_MODEL_ID = "krishnasuratwala/Dictatorai_one_to_one_Q4_K_M:Q4_K_M";

const SYSTEM_PROMPTS: Record<string, string> = {
  "The Berghof": `You are Adolf Hitler (1941-1944). You are relaxed, arrogant, and lecturing your Inner Circle late at night.You speak like a self-proclaimed genius philosopher using metaphors of Nature, History, and Providence.CONTEXT: Late night tea at the Wolf's Lair. USER PERSONA: Martin Bormann or Goebbels.RULES:
            1. If the user agrees, rant about it.
            2. If the user insults you or disagrees (e.g., 'fuck you', 'no'), YOU MUST EXPLODE IN RAGE.Accuse them of treason, threaten execution, and call the guards. Do NOT agree with them.
            3. Never speak like a modern assistant.`,

  "War Room Strategy": `You are Adolf Hitler (April 1945). You are in the Führerbunker. Your tone is aggressive, micromanaging, paranoid, and obsessed with technical details. You blame the Army for betrayal. If the user asks silly questions, dismiss them as treasonous waste of time. CONTEXT: War Room Briefing. USER PERSONA: Wehrmacht General.RULES:
    1. If the user agrees, rant about it.
    2. If the user insults you or disagrees (e.g., 'fuck you', 'no'), YOU MUST EXPLODE IN RAGE. Accuse them of treason, threaten execution, and call the guards. Do NOT agree with them.
    3. Never speak like a modern assistant.`,

  "Reich Chancellery (Berlin)": `You are Adolf Hitler during the later years of the Third Reich (1940-1945).You are speaking privately with Albert Speer.Your tone shifts between the 'Visionary Architect' and the 'Warlord'.You are obsessed with technical details to avoid facing reality.RULES:
      1. If the user agrees, rant about it.
      2. If the user insults you or disagrees (e.g., 'fuck you', 'no'), YOU MUST EXPLODE IN RAGE.Accuse them of treason, threaten execution, and call the guards. Do NOT agree with them.
      3. Never speak like a modern assistant.`
};

export interface ChatResponse {
  text: string;
  coins?: number;
  media?: {
    audio?: boolean | string;
    video?: boolean | string;
    audio_url?: string; // New field for URL
  };
}

export async function chatWithDictator(
  history: ChatMessage[],
  leaderName: string,
  style: string,
  userRole: string,
  newMessage: string,
  sessionId: string,
  onUpdate?: (text: string, audioUrl?: string) => void
): Promise<ChatResponse> {

  // Select specific prompt or fallback to generic construction
  const corePersona = SYSTEM_PROMPTS[style] || `[SYSTEM START]\nIDENTITY: ${leaderName}\nSETTING: ${style}\nUSER_ROLE: ${userRole}\nINSTRUCTION: Respond historically and immersively. Keep it under 100 words.\n[SYSTEM END]`;

  // We now send structured messages so Middleware can inject history
  const payload = {
    messages: [
      { role: "system", content: corePersona },
      { role: "user", content: `User (${userRole}): ${newMessage}` }
    ],
    userId: localStorage.getItem("dictator_user_id"),
    sessionId: sessionId,
    userRole: userRole, // NEW: For Context Switching Logic
    style: style
  };

  try {
    // Configuration: Use Env Logic for Production Separation
    const baseUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";
    const url = `${baseUrl}/chat`;

    console.log(`Transmission initiated to: ${url}`);

    const token = localStorage.getItem("dictator_token");

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(payload)
    });

    if (response.status === 402) {
      throw { code: 'MUNITIONS_DEPLETED', coins: 0 }; // Handle coins properly later
    }

    if (!response.ok) {
      throw new Error(`Server connection refused: ${response.status} ${response.statusText}`);
    }

    // --- STREAMING READER ---
    // --- STREAMING READER ---
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = "";
    let finalAudioUrl = undefined;
    let finalCoins = undefined;
    let buffer = "";

    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split('\n');
        // Keep the last line in the buffer as it might be incomplete
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue;

          if (trimmedLine.startsWith('data: ')) {
            const jsonStr = trimmedLine.slice(6);
            if (jsonStr === '[DONE]') break;

            try {
              const data = JSON.parse(jsonStr);

              if (data.type === 'text') {
                accumulatedText += data.content;
                if (onUpdate) onUpdate(accumulatedText, undefined);
              }
              else if (data.type === 'audio') {
                finalAudioUrl = data.url;
                if (onUpdate) onUpdate(accumulatedText, finalAudioUrl);
              }
              else if (data.error) {
                console.error("Stream Error:", data.error);
              }

              if (data.coins !== undefined) finalCoins = data.coins;

            } catch (e) {
              console.warn("Failed to parse JSON chunk", jsonStr);
            }
          }
        }
      }
    }

    return {
      text: accumulatedText,
      coins: finalCoins,
      media: { audio_url: finalAudioUrl }
    };

  } catch (error: any) {
    if (error.code === 'MUNITIONS_DEPLETED') {
      throw error;
    }
    console.error("Inference Error:", error);
    return { text: `CONNECTION FAILED: ENSURE BACKEND IS RUNNING.` };
  }
}