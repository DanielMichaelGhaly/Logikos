import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { ChatMessage } from '../types/mathviz';

const MessageContainer = styled.div<{ isUser: boolean }>`
  display: flex;
  justify-content: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  margin-bottom: 1.5rem;
  align-items: flex-start;
`;

const Avatar = styled.div`
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, rgba(99,102,241,0.25) 0%, rgba(37,99,235,0.25) 100%);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 1rem;
  font-size: 0.9rem;
  color: #E5E7EB;
  font-weight: 700;
  flex-shrink: 0;
  box-shadow: 0 8px 24px rgba(3, 8, 20, 0.5);
`;

const BubbleContent = styled(motion.div)<{ isUser: boolean }>`
  max-width: 75%;
  background: ${props => props.isUser 
    ? 'linear-gradient(135deg, rgba(37,99,235,0.20), rgba(124,58,237,0.20))'
    : 'rgba(255,255,255,0.92)'
  };
  color: ${props => props.isUser ? '#E5E7EB' : '#111827'};
  padding: 1rem 1.25rem;
  border-radius: ${props => props.isUser 
    ? '18px 18px 8px 18px'
    : '18px 18px 18px 8px'
  };
  font-size: 0.95rem;
  line-height: 1.6;
  box-shadow: ${props => props.isUser
    ? '0 8px 24px rgba(2, 6, 23, 0.5)'
    : '0 4px 14px rgba(2, 6, 23, 0.20)'
  };
  word-wrap: break-word;
  border: ${props => props.isUser ? '1px solid rgba(99,102,241,0.25)' : '1px solid rgba(255,255,255,0.6)'};
`;

const MessageText = styled.div`
  /* Handle markdown-style bold text */
  strong {
    font-weight: 600;
  }
  
  /* Handle code-style text */
  code {
    background: rgba(0, 0, 0, 0.1);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
  }
`;

const Timestamp = styled.div<{ isUser: boolean }>`
  font-size: 0.75rem;
  color: ${props => props.isUser ? 'rgba(255, 255, 255, 0.7)' : '#9ca3af'};
  margin-top: 0.5rem;
  text-align: ${props => props.isUser ? 'right' : 'left'};
`;

interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.type === 'user';
  const avatar = message.has_solution ? "MV" : "AI";

  const formatContent = (content: string): string | null => {
    // Parse LaTeX expressions and markdown
    let remaining = content;

    // First handle LaTeX expressions (looking for patterns like 2x^2, f'(x), etc.)
    const mathPatterns = [
      /\b\d*[a-zA-Z]\^?\{?[0-9a-zA-Z]*\}?/g, // Variables with exponents like x^2, x^{2}
      /\b[a-zA-Z]'?\([a-zA-Z]\)/g, // Functions like f(x), f'(x)
      /\b\d+[a-zA-Z]\s*[+\-]\s*\d+/g, // Terms like 2x + 3
      /\b[a-zA-Z]\s*=\s*[0-9\-+*/^()a-zA-Z\s]+/g, // Equations like x = 2
      /∫.*?dx/g, // Integrals
      /∑.*?/g, // Summations
      /√\(.*?\)/g, // Square roots
    ];

    let hasMatches = false;

    for (const pattern of mathPatterns) {
      const matches = Array.from(remaining.matchAll(pattern));
      if (matches.length > 0) {
        hasMatches = true;
        break;
      }
    }

    if (!hasMatches) {
      // No math expressions found, just handle markdown
      return remaining
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
    }

    // If we have math, we need to render as JSX components
    return null; // Signal to use JSX rendering
  };

  const renderContentWithMath = (content: string) => {
    const parts: React.ReactNode[] = [];
    let remaining = content;
    let index = 0;

    // Simple math detection patterns
    const mathRegex = /(\b\d*[a-zA-Z]\^?\{?[0-9a-zA-Z]*\}?|\b[a-zA-Z]'?\([a-zA-Z]\)|\b\d+[a-zA-Z]\s*[+\-]\s*\d+|\b[a-zA-Z]\s*=\s*[0-9\-+*/^()a-zA-Z\s]+|∫.*?dx|∑.*?|√\(.*?\))/g;

    let match;
    let lastEnd = 0;

    while ((match = mathRegex.exec(remaining)) !== null) {
      // Add text before the match
      if (match.index > lastEnd) {
        const textBefore = remaining.slice(lastEnd, match.index);
        if (textBefore.trim()) {
          parts.push(
            <span key={index++} dangerouslySetInnerHTML={{
              __html: textBefore
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>')
            }} />
          );
        }
      }

      // Add the math expression
      try {
        parts.push(<InlineMath key={index++}>{match[1]}</InlineMath>);
      } catch (error) {
        // Fallback to plain text if LaTeX fails
        parts.push(<code key={index++}>{match[1]}</code>);
      }

      lastEnd = match.index + match[0].length;
    }

    // Add remaining text
    if (lastEnd < remaining.length) {
      const textAfter = remaining.slice(lastEnd);
      if (textAfter.trim()) {
        parts.push(
          <span key={index++} dangerouslySetInnerHTML={{
            __html: textAfter
              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              .replace(/`(.*?)`/g, '<code>$1</code>')
              .replace(/\n/g, '<br>')
          }} />
        );
      }
    }

    return parts.length > 0 ? parts : [content];
  };

  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <MessageContainer isUser={isUser}>
      {!isUser && <Avatar>{avatar}</Avatar>}
      
      <BubbleContent
        isUser={isUser}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ 
          type: "spring",
          stiffness: 500,
          damping: 30
        }}
      >
        <MessageText>
          {(() => {
            const formattedContent = formatContent(message.content);
            return formattedContent === null
              ? renderContentWithMath(message.content)
              : <span dangerouslySetInnerHTML={{ __html: formattedContent }} />;
          })()}
        </MessageText>
        <Timestamp isUser={isUser}>
          {formatTime(message.timestamp)}
        </Timestamp>
      </BubbleContent>
    </MessageContainer>
  );
};