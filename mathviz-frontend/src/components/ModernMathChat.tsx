import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// Types
interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  visualization?: string;
  isLoading?: boolean;
}

// Styled Components
const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #000000;
  color: #FFFFFF;
`;

const Header = styled.div`
  padding: 20px 24px;
  border-bottom: 1px solid #1A1A1A;
  background: #000000;
`;

const Title = styled.h1`
  font-size: 24px;
  font-weight: 600;
  color: #FFFFFF;
  margin: 0;
`;

const Subtitle = styled.p`
  font-size: 14px;
  color: #888888;
  margin: 4px 0 0 0;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 0;
  display: flex;
  flex-direction: column;

  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: transparent;
  }

  &::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #444444;
  }
`;

const MessageGroup = styled(motion.div)<{ $isUser: boolean }>`
  width: 100%;
  padding: 24px;
  display: flex;
  justify-content: ${props => props.$isUser ? 'flex-end' : 'flex-start'};
`;

const MessageBubble = styled.div<{ $isUser: boolean }>`
  max-width: 70%;
  min-width: 200px;
  padding: 16px 20px;
  border-radius: 18px;
  font-size: 15px;
  line-height: 1.5;
  word-wrap: break-word;

  ${props => props.$isUser ? `
    background: #FFFFFF;
    color: #000000;
    border-bottom-right-radius: 4px;
  ` : `
    background: #1A1A1A;
    color: #FFFFFF;
    border-bottom-left-radius: 4px;
    border: 1px solid #2A2A2A;
  `}
`;

const VisualizationContainer = styled.div`
  margin-top: 16px;
  padding: 20px;
  background: #0F0F0F;
  border-radius: 12px;
  border: 1px solid #2A2A2A;
  overflow: auto;

  /* Ensure visualizations display properly */
  img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
  }

  /* Style embedded HTML content */
  iframe {
    width: 100%;
    border: none;
    border-radius: 8px;
  }
`;

const InputContainer = styled.div`
  padding: 24px;
  border-top: 1px solid #1A1A1A;
  background: #000000;
`;

const InputWrapper = styled.div`
  display: flex;
  align-items: flex-end;
  gap: 12px;
  max-width: 800px;
  margin: 0 auto;
`;

const MessageInput = styled.textarea`
  flex: 1;
  min-height: 20px;
  max-height: 120px;
  padding: 12px 16px;
  border: 1px solid #333333;
  border-radius: 12px;
  background: #1A1A1A;
  color: #FFFFFF;
  font-family: inherit;
  font-size: 15px;
  line-height: 1.4;
  resize: none;
  outline: none;

  &::placeholder {
    color: #666666;
  }

  &:focus {
    border-color: #FFFFFF;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1);
  }
`;

const SendButton = styled.button<{ $disabled: boolean }>`
  padding: 12px 16px;
  border: none;
  border-radius: 12px;
  background: ${props => props.$disabled ? '#333333' : '#FFFFFF'};
  color: ${props => props.$disabled ? '#666666' : '#000000'};
  font-weight: 600;
  cursor: ${props => props.$disabled ? 'not-allowed' : 'pointer'};
  transition: all 0.15s ease;

  &:hover {
    background: ${props => props.$disabled ? '#333333' : '#E5E5E5'};
  }
`;

const LoadingDots = styled.div`
  display: flex;
  gap: 4px;
  padding: 8px 0;

  span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #666666;
    animation: loadingDot 1.5s infinite ease-in-out;

    &:nth-child(1) { animation-delay: 0s; }
    &:nth-child(2) { animation-delay: 0.2s; }
    &:nth-child(3) { animation-delay: 0.4s; }
  }

  @keyframes loadingDot {
    0%, 60%, 100% { opacity: 0.3; }
    30% { opacity: 1; }
  }
`;

const ExamplePrompts = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
`;

const ExamplePrompt = styled.button`
  padding: 8px 12px;
  border: 1px solid #333333;
  border-radius: 8px;
  background: transparent;
  color: #CCCCCC;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s ease;

  &:hover {
    border-color: #FFFFFF;
    color: #FFFFFF;
  }
`;

// Helper function to render math in text
const renderMathText = (text: string) => {
  // Split by math delimiters and render accordingly
  const parts = text.split(/(\$\$[\s\S]*?\$\$|\$[\s\S]*?\$)/);

  return parts.map((part, index) => {
    if (part.startsWith('$$') && part.endsWith('$$')) {
      // Block math
      return <BlockMath key={index}>{part.slice(2, -2)}</BlockMath>;
    } else if (part.startsWith('$') && part.endsWith('$')) {
      // Inline math
      return <InlineMath key={index}>{part.slice(1, -1)}</InlineMath>;
    } else {
      // Regular text
      return <span key={index}>{part}</span>;
    }
  });
};

export const ModernMathChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'assistant',
      content: "Hello! I'm your AI math tutor. I can solve equations, create visualizations, and explain mathematical concepts. Try asking me about contour maps like 'x^2 + y^2' or any other math problem!",
      timestamp: new Date(),
    }
  ]);

  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Add loading message
    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages(prev => [...prev, loadingMessage]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          include_reasoning: true,
          include_visualization: true,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      console.log('API Response:', data);

      // Remove loading message and add real response
      setMessages(prev => {
        const filtered = prev.filter(m => !m.isLoading);

        const assistantMessage: ChatMessage = {
          id: (Date.now() + 2).toString(),
          type: 'assistant',
          content: data.reply_text || 'I received your question and processed it successfully.',
          timestamp: new Date(),
          visualization: data.solution?.visualization || data.solution?.metadata?.graph_html,
        };

        return [...filtered, assistantMessage];
      });

    } catch (error) {
      console.error('Error:', error);

      // Remove loading message and add error message
      setMessages(prev => {
        const filtered = prev.filter(m => !m.isLoading);

        const errorMessage: ChatMessage = {
          id: (Date.now() + 3).toString(),
          type: 'assistant',
          content: 'Sorry, I encountered an error processing your request. Please try again.',
          timestamp: new Date(),
        };

        return [...filtered, errorMessage];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleExamplePrompt = (prompt: string) => {
    setInputValue(prompt);
    inputRef.current?.focus();
  };

  return (
    <ChatContainer>
      <Header>
        <Title>Logikos Math AI</Title>
        <Subtitle>Advanced mathematical problem solving with visualizations</Subtitle>
      </Header>

      <MessagesContainer>
        <AnimatePresence>
          {messages.map((message) => (
            <MessageGroup
              key={message.id}
              $isUser={message.type === 'user'}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <MessageBubble $isUser={message.type === 'user'}>
                {message.isLoading ? (
                  <LoadingDots>
                    <span></span>
                    <span></span>
                    <span></span>
                  </LoadingDots>
                ) : (
                  <>
                    <div>{renderMathText(message.content)}</div>
                    {message.visualization && (
                      <VisualizationContainer>
                        <div dangerouslySetInnerHTML={{ __html: message.visualization }} />
                      </VisualizationContainer>
                    )}
                  </>
                )}
              </MessageBubble>
            </MessageGroup>
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputContainer>
        <InputWrapper>
          <MessageInput
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about math problems, equations, or visualizations..."
            disabled={isLoading}
          />
          <SendButton
            $disabled={!inputValue.trim() || isLoading}
            onClick={handleSubmit}
          >
            Send
          </SendButton>
        </InputWrapper>

        {messages.length === 1 && (
          <ExamplePrompts>
            <ExamplePrompt onClick={() => handleExamplePrompt('x^2 + y^2')}>
              x² + y² contour
            </ExamplePrompt>
            <ExamplePrompt onClick={() => handleExamplePrompt('sin(x)*cos(y)')}>
              sin(x)cos(y) pattern
            </ExamplePrompt>
            <ExamplePrompt onClick={() => handleExamplePrompt('solve 2x + 5 = 0')}>
              solve equation
            </ExamplePrompt>
            <ExamplePrompt onClick={() => handleExamplePrompt('derivative of x^3')}>
              find derivative
            </ExamplePrompt>
          </ExamplePrompts>
        )}
      </InputContainer>
    </ChatContainer>
  );
};