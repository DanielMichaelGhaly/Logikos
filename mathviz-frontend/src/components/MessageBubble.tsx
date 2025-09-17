import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
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

  const formatContent = (content: string) => {
    // Simple markdown-style formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
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
        <MessageText 
          dangerouslySetInnerHTML={{ 
            __html: formatContent(message.content) 
          }} 
        />
        <Timestamp isUser={isUser}>
          {formatTime(message.timestamp)}
        </Timestamp>
      </BubbleContent>
    </MessageContainer>
  );
};