import React, { useState, useRef, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatMessage, MathSolution, APIResponse } from '../types/mathviz';
import { MessageBubble } from './MessageBubble';
import { SolutionDisplay } from './SolutionDisplay';
import { DesmosGraph } from './DesmosGraph';
import { mathvizAPI } from '../services/api';

const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const ChatContainer = styled.div`
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 1.25rem;
  height: 100vh;
  padding: 1.25rem;
font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
  position: relative;
`;

const Header = styled(motion.div)`
  background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 24px;
  padding: 2rem;
  margin: 1rem 1rem 1rem 0rem;
  text-align: center;
  box-shadow: 0 20px 50px rgba(3, 8, 20, 0.6);
  animation: ${fadeInUp} 0.8s ease-out;
`;

const Title = styled.h1`
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.015em;
  color: #E5E7EB;
  margin-bottom: 0.25rem;
`;

const Subtitle = styled.p`
  font-size: 0.95rem;
  color: rgba(226, 232, 240, 0.65);
  font-weight: 500;
  margin: 0;
`;

const MessagesContainer = styled.div`
  flex: 1;
  background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 20px;
  margin: 0 1rem 0 0rem;
  padding: 2rem;
  overflow-y: auto;
  box-shadow: 0 20px 50px rgba(3, 8, 20, 0.6);
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
  }
`;

const InputArea = styled.div`
  background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 20px;
  margin: 1rem 1rem 1rem 0rem;
  padding: 2rem;
  box-shadow: 0 20px 50px rgba(3, 8, 20, 0.6);
`;

const SuggestionsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1.5rem;
`;

const SuggestionChip = styled(motion.button)`
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(56, 189, 248, 0.10));
  border: 1px solid rgba(99, 102, 241, 0.25);
  color: #93C5FD;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(56, 189, 248, 0.16));
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.30);
  }
`;

const InputRow = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
`;

const MathInput = styled.input`
  flex: 1;
  padding: 1rem 1.5rem;
  background: rgba(255,255,255,0.06);
  color: #E5E7EB;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 12px;
  font-size: 1rem;
  font-family: 'JetBrains Mono', monospace;
  transition: all 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: rgba(99, 102, 241, 0.6);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18);
  }
  
  &::placeholder {
    color: rgba(226,232,240,0.6);
    font-style: italic;
  }
`;

const SendButton = styled(motion.button)<{ disabled: boolean }>`
  background: ${props => props.disabled 
    ? 'linear-gradient(135deg, #475569 0%, #334155 100%)'
    : 'linear-gradient(135deg, #2563EB 0%, #7C3AED 100%)'
  };
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-weight: 700;
  letter-spacing: 0.01em;
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    transform: ${props => props.disabled ? 'none' : 'translateY(-1px)'};
    box-shadow: ${props => props.disabled ? 'none' : '0 8px 24px rgba(59, 130, 246, 0.35)'};
  }
`;

const TypingIndicator = styled(motion.div)`
  display: flex;
  align-items: center;
  padding: 1rem 1.25rem;
  background: #f3f4f6;
  border-radius: 20px 20px 20px 8px;
  margin: 1rem 0;
  color: #6b7280;
  font-size: 0.9rem;
`;

const TypingDot = styled(motion.span)`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #6b7280;
  margin: 0 2px;
`;

const suggestions = [
  "Solve: 2x² - 8x + 6 = 0",
  "Find derivative of x³ + 4x² - 2x + 1",
  "Integrate: ∫(3x² + 2x + 1) dx",
  "Factor: x² - 5x + 6",
  "Graph: y = x² - 4x + 3",
  "Find critical points of x³ - 3x² + 2"
];

const Sidebar = styled.div`
  height: calc(100vh - 2.5rem);
  background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 20px;
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  box-shadow: 0 20px 50px rgba(3, 8, 20, 0.6);
`;

const NavSection = styled.div`
  margin-top: 0.5rem;
`;

const NavItem = styled.button`
  width: 100%;
  text-align: left;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  color: #E5E7EB;
  padding: 0.75rem 0.9rem;
  border-radius: 12px;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  cursor: pointer;
  
  &:hover {
    background: rgba(255,255,255,0.10);
    border-color: rgba(255,255,255,0.22);
  }
`;

const Brand = styled.div`
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  background: linear-gradient(135deg, #A5B4FC 0%, #60A5FA 30%, #22D3EE 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 1.25rem;
`;

export const MathVizChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      content: "👋 Hello! I'm your AI-powered math tutor. I can solve equations, find derivatives, compute integrals, and create interactive graphs. What would you like to learn today?",
      timestamp: new Date(),
      has_solution: false
    }
  ]);
  
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: text,
      timestamp: new Date(),
      has_solution: false
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call MathViz API
      const response = await mathvizAPI.chat(text);

      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response.reply_text || `I'll solve that now: **${text}**`,
        timestamp: new Date(),
        solution: response.solution,
        desmos_url: response.desmos_url,
        desmos_config: response.desmos_config,
        has_solution: !!response.solution
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: `🤔 I encountered an issue solving that problem: **${error instanceof Error ? error.message : 'Unknown error'}**\n\nCould you try rephrasing it or asking a different question? I'm here to help!`,
        timestamp: new Date(),
        has_solution: false
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
  };

  const handleInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage(inputValue);
  };

  const showSuggestions = messages.length <= 1;

  return (
    <ChatContainer>
      {/* Sidebar */}
      <Sidebar>
        <Brand>∞ Logikos</Brand>
        <NavSection>
          <NavItem>💬 Chats</NavItem>
        </NavSection>
        <div style={{ flex: 1 }} />
        <NavSection>
          <NavItem>⚙️ Settings</NavItem>
          <NavItem>↩️ Log Out</NavItem>
        </NavSection>
      </Sidebar>

      {/* Main Panel */}
      <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Title>Logikos</Title>
          <Subtitle>Your AI-powered mathematical companion with step-by-step solutions</Subtitle>
        </Header>

        <MessagesContainer>
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
              >
                <MessageBubble message={message} />
                {message.solution && (
                  <SolutionDisplay solution={message.solution} />
                )}
                {(message.desmos_url || message.desmos_config) && (
                  <DesmosGraph 
                    url={message.desmos_url}
                    expressions={message.desmos_config?.expressions}
                    viewport={{
                      xmin: message.desmos_config?.xmin,
                      xmax: message.desmos_config?.xmax,
                      ymin: message.desmos_config?.ymin,
                      ymax: message.desmos_config?.ymax,
                    }}
                  />
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <TypingIndicator
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
            <span>Solving your problem…</span>
              {[0, 1, 2].map(i => (
                <TypingDot
                  key={i}
                  animate={{ 
                    scale: [0.8, 1.2, 0.8],
                    opacity: [0.5, 1, 0.5]
                  }}
                  transition={{
                    duration: 1.4,
                    repeat: Infinity,
                    delay: i * 0.2
                  }}
                />
              ))}
            </TypingIndicator>
          )}
          
          <div ref={messagesEndRef} />
        </MessagesContainer>

        <InputArea>
          {showSuggestions && (
            <div>
              <h3 style={{ marginBottom: '1rem', color: 'rgba(226,232,240,0.85)', fontWeight: 700 }}>
                💡 Try these examples:
              </h3>
              <SuggestionsGrid>
                {suggestions.map((suggestion, index) => (
                  <SuggestionChip
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {suggestion}
                  </SuggestionChip>
                ))}
              </SuggestionsGrid>
            </div>
          )}

          <form onSubmit={handleInputSubmit}>
            <InputRow>
              <MathInput
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Type your math problem here... (e.g., 'Solve: 2x² - 8x + 6 = 0')"
                disabled={isLoading}
              />
              <SendButton
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                whileHover={{ scale: isLoading ? 1 : 1.05 }}
                whileTap={{ scale: isLoading ? 1 : 0.95 }}
              >
              {isLoading ? '…' : 'Send'}
              </SendButton>
            </InputRow>
          </form>
        </InputArea>
      </div>
    </ChatContainer>
  );
};