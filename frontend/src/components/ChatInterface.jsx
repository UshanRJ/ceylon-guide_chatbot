// frontend/src/components/ChatInterface.jsx - ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState, useEffect, useRef } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { FiSend, FiLoader, FiTrash2, FiDownload, FiMoreVertical, FiMessageSquare } from 'react-icons/fi';
import chatService from '../services/chatService';
import WeatherWidget from './tools/WeatherWidget';
import CurrencyConverter from './tools/CurrencyConverter';
import TranslatorWidget from './tools/TranslatorWidget';
import MapWidget from './tools/MapWidget';

// Enhanced Animations
const fadeIn = keyframes`
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
`;

const slideUp = keyframes`
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
`;

const messageSlideIn = keyframes`
    from {
        opacity: 0;
        transform: translateX(-20px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
`;

const userMessageSlideIn = keyframes`
    from {
        opacity: 0;
        transform: translateX(20px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
`;

const pulse = keyframes`
    0%, 100% {
        transform: scale(1);
        opacity: 0.8;
    }
    50% {
        transform: scale(1.02);
        opacity: 1;
    }
`;

const shimmer = keyframes`
    0% {
        background-position: -200px 0;
    }
    100% {
        background-position: calc(200px + 100%) 0;
    }
`;

const bounce = keyframes`
    0%, 20%, 60%, 100% {
        transform: translateY(0);
    }
    40% {
        transform: translateY(-3px);
    }
    80% {
        transform: translateY(-1px);
    }
`;

const floatingGlow = keyframes`
    0%, 100% {
        box-shadow: 0 2px 20px rgba(102, 126, 234, 0.1);
    }
    50% {
        box-shadow: 0 4px 30px rgba(102, 126, 234, 0.2);
    }
`;

// Enhanced Container with Glass Morphism
const Container = styled.div`
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.9) 0%, 
        rgba(247, 250, 252, 0.9) 100%
    );
    backdrop-filter: blur(20px);
    position: relative;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 30%, rgba(102, 126, 234, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(118, 75, 162, 0.03) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        height: calc(100vh - 60px);
        margin-top: 60px;
    }
`;

// Enhanced Header with Better Visual Hierarchy
const Header = styled.header`
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(226, 232, 240, 0.5);
    padding: 1.5rem 2rem;
    position: relative;
    z-index: 10;
    animation: ${floatingGlow} 6s ease-in-out infinite;

    &::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 10%;
        right: 10%;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(102, 126, 234, 0.3) 50%, 
            transparent 100%
        );
    }

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        display: none;
    }
`;

// Enhanced Title with Actions
const HeaderContent = styled.div`
    display: flex;
    align-items: center;
    justify-content: space-between;
`;

const Title = styled.h1`
    font-size: 1.6rem;
    color: ${props => props.theme.colors.text.primary};
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    .icon {
        font-size: 1.4rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: ${bounce} 2s ease-in-out infinite;
    }
`;

// Enhanced Header Actions
const HeaderActions = styled.div`
    display: flex;
    align-items: center;
    gap: 0.75rem;
`;

const ActionButton = styled.button`
    padding: 0.5rem;
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 8px;
    color: ${props => props.theme.colors.primary};
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(102, 126, 234, 0.2) 50%, 
            transparent 100%
        );
        transition: left 0.5s ease;
    }

    &:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);

        &::before {
            left: 100%;
        }
    }

    &:active {
        transform: scale(0.95);
    }

    &.danger {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;

        &:hover {
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.3);
        }
    }
`;

// Enhanced Chat Area
const ChatArea = styled.div`
    flex: 1;
    display: flex;
    overflow: hidden;
    position: relative;
    z-index: 1;

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        flex-direction: column;
    }
`;

// Enhanced Messages Section
const MessagesSection = styled.div`
    flex: 1;
    display: flex;
    flex-direction: column;
    background: rgba(248, 250, 252, 0.5);
    backdrop-filter: blur(10px);
    position: relative;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23667eea' fill-opacity='0.02'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
    }
`;

// Enhanced Messages Container
const MessagesContainer = styled.div`
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    position: relative;
    z-index: 1;

    /* Enhanced scrollbar */
    &::-webkit-scrollbar {
        width: 6px;
    }

    &::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 3px;
    }

    &::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 3px;
        transition: all 0.3s ease;
    }

    &::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #5a67d8 0%, #6b46c1 100%);
    }
`;

// Enhanced Message Wrapper
const MessageWrapper = styled.div`
    display: flex;
    justify-content: ${props => props.$isUser ? 'flex-end' : 'flex-start'};
    margin-bottom: 0.5rem;
    animation: ${props => props.$isUser ? userMessageSlideIn : messageSlideIn} 0.3s ease-out;
`;

// Enhanced Message Bubble
const Message = styled.div`
    max-width: 75%;
    padding: 1rem 1.25rem;
    border-radius: 18px;
    word-wrap: break-word;
    position: relative;
    backdrop-filter: blur(10px);
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;

    &:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    }

    ${props => props.$isUser ? css`
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 6px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);

        &::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.1) 0%, 
                transparent 50%
            );
            border-radius: inherit;
            pointer-events: none;
        }
    ` : css`
        background: rgba(255, 255, 255, 0.9);
        color: ${props => props.theme.colors.text.primary};
        border: 1px solid rgba(226, 232, 240, 0.5);
        border-bottom-left-radius: 6px;

        &::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, 
                rgba(102, 126, 234, 0.02) 0%, 
                transparent 50%
            );
            border-radius: inherit;
            pointer-events: none;
        }
    `}

    @media (max-width: ${props => props.theme.breakpoints.mobile}) {
        max-width: 90%;
    }
`;

// Enhanced Timestamp
const Timestamp = styled.span`
    font-size: 0.75rem;
    color: ${props => props.$isUser ? 'rgba(255, 255, 255, 0.8)' : props.theme.colors.text.tertiary};
    margin-top: 0.5rem;
    display: block;
    font-weight: 500;
`;

// Enhanced Input Container
const InputContainer = styled.form`
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px);
    border-top: 1px solid rgba(226, 232, 240, 0.5);
    display: flex;
    gap: 1rem;
    align-items: center;
    position: relative;
    z-index: 10;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 10%;
        right: 10%;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(102, 126, 234, 0.3) 50%, 
            transparent 100%
        );
    }
`;

// Enhanced Input Field
const Input = styled.input`
    flex: 1;
    padding: 0.875rem 1.25rem;
    border: 2px solid rgba(226, 232, 240, 0.5);
    border-radius: 12px;
    font-size: 1rem;
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;

    &::placeholder {
        color: ${props => props.theme.colors.text.tertiary};
    }

    &:focus {
        outline: none;
        border-color: ${props => props.theme.colors.primary};
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        transform: translateY(-1px);
    }

    &:disabled {
        background: rgba(241, 245, 249, 0.8);
        cursor: not-allowed;
    }
`;

// Enhanced Send Button
const SendButton = styled.button`
    padding: 0.875rem 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(255, 255, 255, 0.2) 50%, 
            transparent 100%
        );
        transition: left 0.5s ease;
    }

    &:hover:not(:disabled) {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);

        &::before {
            left: 100%;
        }
    }

    &:active:not(:disabled) {
        transform: translateY(0);
    }

    &:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
    }
`;

// Enhanced Quick Actions
const QuickActions = styled.div`
    padding: 1rem 1.5rem;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(15px);
    border-top: 1px solid rgba(226, 232, 240, 0.3);
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 10;

    @media (max-width: ${props => props.theme.breakpoints.mobile}) {
        padding: 1rem;
        gap: 0.5rem;
    }
`;

// Enhanced Quick Action Button
const QuickActionButton = styled.button`
    padding: 0.625rem 1rem;
    background: rgba(102, 126, 234, 0.08);
    color: ${props => props.theme.colors.primary};
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    white-space: nowrap;
    position: relative;
    overflow: hidden;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(102, 126, 234, 0.1) 50%, 
            transparent 100%
        );
        transition: left 0.5s ease;
    }

    &:hover:not(:disabled) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);

        &::before {
            left: 100%;
        }
    }

    &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
`;

// Enhanced Tool Panel
const ToolPanel = styled.div`
    width: 380px;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px);
    border-left: 1px solid rgba(226, 232, 240, 0.5);
    overflow-y: auto;
    animation: ${slideUp} 0.3s ease-out;

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        width: 100%;
        border-left: none;
        border-top: 1px solid rgba(226, 232, 240, 0.5);
    }
`;

// Enhanced Loading Message
const LoadingMessage = styled.div`
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: ${props => props.theme.colors.text.secondary};
    font-style: italic;
    font-weight: 500;

    svg {
        animation: spin 1s linear infinite;
        color: ${props => props.theme.colors.primary};
    }

    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;

// Enhanced Welcome Message
const WelcomeMessage = styled.div`
    text-align: center;
    padding: 3rem 2rem;
    color: ${props => props.theme.colors.text.secondary};
    animation: ${fadeIn} 0.6s ease-out;
    position: relative;

    &::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.05) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        animation: ${pulse} 4s ease-in-out infinite;
    }

    h2 {
        font-size: 1.75rem;
        color: ${props => props.theme.colors.text.primary};
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }

    p {
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }

    .emoji {
        font-size: 2rem;
        margin-bottom: 1rem;
        display: block;
        animation: ${bounce} 2s ease-in-out infinite;
    }
`;

// Confirmation Modal
const ModalOverlay = styled.div`
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    opacity: ${props => props.$show ? 1 : 0};
    pointer-events: ${props => props.$show ? 'all' : 'none'};
    transition: all 0.3s ease;
`;

const Modal = styled.div`
    background: white;
    border-radius: 16px;
    padding: 2rem;
    max-width: 400px;
    width: 90%;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    transform: ${props => props.$show ? 'scale(1)' : 'scale(0.9)'};
    transition: all 0.3s ease;

    h3 {
        margin-bottom: 1rem;
        color: ${props => props.theme.colors.text.primary};
        font-size: 1.25rem;
        font-weight: 600;
    }

    p {
        margin-bottom: 1.5rem;
        color: ${props => props.theme.colors.text.secondary};
        line-height: 1.5;
    }

    .modal-actions {
        display: flex;
        gap: 1rem;
        justify-content: flex-end;

        button {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;

            &.cancel {
                background: transparent;
                border: 1px solid ${props => props.theme.colors.border.medium};
                color: ${props => props.theme.colors.text.secondary};

                &:hover {
                    background: ${props => props.theme.colors.background.secondary};
                }
            }

            &.confirm {
                background: #ef4444;
                border: 1px solid #ef4444;
                color: white;

                &:hover {
                    background: #dc2626;
                }
            }
        }
    }
`;

// Enhanced Chat Interface Component
const ChatInterface = ({ activeView }) => {
    // YOUR ORIGINAL STATE - COMPLETELY PRESERVED
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    // NEW STATE FOR ENHANCED FEATURES
    const [showClearModal, setShowClearModal] = useState(false);

    // YOUR ORIGINAL QUICK ACTIONS - ENHANCED
    const quickActions = [
        "What's the weather in Colombo?",
        "Convert 100 USD to LKR",
        "Translate 'Hello' to Sinhala",
        "Best beaches in Sri Lanka",
        "Show me Sigiriya on map"
    ];

    // YOUR ORIGINAL useEffect - COMPLETELY PRESERVED
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // YOUR ORIGINAL SCROLL FUNCTION - COMPLETELY PRESERVED
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // YOUR ORIGINAL SUBMIT HANDLER - COMPLETELY PRESERVED
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!inputValue.trim() || isLoading) return;

        const userMessage = {
            id: Date.now(),
            type: 'user',
            text: inputValue.trim(),
            timestamp: new Date().toLocaleTimeString()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);

        try {
            const response = await chatService.sendMessage(userMessage.text);

            const botMessage = {
                id: Date.now() + 1,
                type: 'bot',
                text: response.response,
                timestamp: new Date().toLocaleTimeString()
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            const errorMessage = {
                id: Date.now() + 1,
                type: 'bot',
                text: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date().toLocaleTimeString()
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    // YOUR ORIGINAL QUICK ACTION HANDLER - COMPLETELY PRESERVED
    const handleQuickAction = (action) => {
        setInputValue(action);
    };

    // YOUR ORIGINAL TOOL RENDERER - COMPLETELY PRESERVED
    const renderTool = () => {
        switch (activeView) {
            case 'weather':
                return <WeatherWidget />;
            case 'currency':
                return <CurrencyConverter />;
            case 'translate':
                return <TranslatorWidget />;
            case 'map':
                return <MapWidget />;
            default:
                return null;
        }
    };

    // NEW FEATURE: Clear chat functionality
    const handleClearChat = () => {
        setShowClearModal(true);
    };

    const confirmClearChat = () => {
        setMessages([]);
        setShowClearModal(false);
        // Add haptic feedback if available
        if (navigator.vibrate) {
            navigator.vibrate(50);
        }
    };

    // NEW FEATURE: Download chat functionality
    const handleDownloadChat = () => {
        if (messages.length === 0) {
            alert('No messages to download!');
            return;
        }

        const chatContent = messages
            .map(msg => `[${msg.timestamp}] ${msg.type === 'user' ? 'You' : 'Ceylon Guide'}: ${msg.text}`)
            .join('\n\n');

        const blob = new Blob([chatContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ceylon-guide-chat-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Add haptic feedback if available
        if (navigator.vibrate) {
            navigator.vibrate([50, 100, 50]);
        }
    };

    return (
        <Container>
            <Header>
                <HeaderContent>
                    <Title>
                        <FiMessageSquare className="icon" />
                        Ceylon Guide Chat Assistant
                    </Title>
                    <HeaderActions>
                        <ActionButton
                            onClick={handleDownloadChat}
                            disabled={messages.length === 0}
                            title="Download chat history"
                            aria-label="Download chat history"
                        >
                            <FiDownload />
                        </ActionButton>
                        <ActionButton
                            onClick={handleClearChat}
                            disabled={messages.length === 0}
                            className="danger"
                            title="Clear chat history"
                            aria-label="Clear chat history"
                        >
                            <FiTrash2 />
                        </ActionButton>
                    </HeaderActions>
                </HeaderContent>
            </Header>

            <ChatArea>
                <MessagesSection>
                    <MessagesContainer>
                        {messages.length === 0 ? (
                            <WelcomeMessage>
                                <span className="emoji">🇱🇰</span>
                                <h2>Welcome to Ceylon Guide!</h2>
                                <p>I'm here to help you explore the beautiful island of Sri Lanka.</p>
                                <p>Ask me about weather, currency conversions, translations, or amazing places to visit!</p>
                                <p>Let's start your Sri Lankan adventure together! ✨</p>
                            </WelcomeMessage>
                        ) : (
                            messages.map((message) => (
                                <MessageWrapper key={message.id} $isUser={message.type === 'user'}>
                                    <Message $isUser={message.type === 'user'}>
                                        {message.text}
                                        <Timestamp $isUser={message.type === 'user'}>
                                            {message.timestamp}
                                        </Timestamp>
                                    </Message>
                                </MessageWrapper>
                            ))
                        )}

                        {isLoading && (
                            <MessageWrapper $isUser={false}>
                                <Message $isUser={false}>
                                    <LoadingMessage>
                                        <FiLoader />
                                        Ceylon Guide is thinking...
                                    </LoadingMessage>
                                </Message>
                            </MessageWrapper>
                        )}

                        <div ref={messagesEndRef} />
                    </MessagesContainer>

                    <QuickActions>
                        {quickActions.map((action, index) => (
                            <QuickActionButton
                                key={index}
                                onClick={() => handleQuickAction(action)}
                                disabled={isLoading}
                                title={`Quick action: ${action}`}
                            >
                                {action}
                            </QuickActionButton>
                        ))}
                    </QuickActions>

                    <InputContainer onSubmit={handleSubmit}>
                        <Input
                            type="text"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            placeholder="Type your message about Sri Lanka..."
                            disabled={isLoading}
                            autoComplete="off"
                        />
                        <SendButton
                            type="submit"
                            disabled={isLoading || !inputValue.trim()}
                            aria-label="Send message"
                        >
                            <FiSend />
                            Send
                        </SendButton>
                    </InputContainer>
                </MessagesSection>

                {activeView !== 'chat' && (
                    <ToolPanel>
                        {renderTool()}
                    </ToolPanel>
                )}
            </ChatArea>

            {/* NEW FEATURE: Clear Chat Confirmation Modal */}
            <ModalOverlay
                $show={showClearModal}
                onClick={() => setShowClearModal(false)}
            >
                <Modal
                    $show={showClearModal}
                    onClick={(e) => e.stopPropagation()}
                >
                    <h3>Clear Chat History</h3>
                    <p>
                        Are you sure you want to clear all chat messages?
                        This action cannot be undone.
                    </p>
                    <div className="modal-actions">
                        <button
                            className="cancel"
                            onClick={() => setShowClearModal(false)}
                        >
                            Cancel
                        </button>
                        <button
                            className="confirm"
                            onClick={confirmClearChat}
                        >
                            Clear Chat
                        </button>
                    </div>
                </Modal>
            </ModalOverlay>
        </Container>
    );
};

export default ChatInterface;