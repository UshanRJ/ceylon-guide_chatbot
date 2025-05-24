// frontend/src/components/MobileHeader.jsx - ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState, useEffect } from 'react';
import styled, { keyframes, css } from 'styled-components';

// Enhanced Animations with Modern Easing
const slideDown = keyframes`
    from {
        transform: translateY(-100%);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
`;

const pulse = keyframes`
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
    }
    50% {
        transform: scale(1.02);
        box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
    }
`;

const bounce = keyframes`
    0%, 20%, 60%, 100% {
        transform: translateY(0) rotate(0deg);
    }
    40% {
        transform: translateY(-3px) rotate(-2deg);
    }
    80% {
        transform: translateY(-1px) rotate(1deg);
    }
`;

const gradientShift = keyframes`
    0%, 100% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
`;

const rippleEffect = keyframes`
    0% {
        transform: scale(0);
        opacity: 0.8;
    }
    100% {
        transform: scale(3);
        opacity: 0;
    }
`;

const shimmerPass = keyframes`
    0% {
        transform: translateX(-150px);
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
    100% {
        transform: translateX(150px);
        opacity: 0;
    }
`;

const breathe = keyframes`
    0%, 100% {
        transform: scale(1);
        opacity: 0.6;
    }
    50% {
        transform: scale(1.1);
        opacity: 0.8;
    }
`;

const floatingGlow = keyframes`
    0%, 100% {
        box-shadow:
                0 2px 20px rgba(102, 126, 234, 0.2),
                0 0 40px rgba(118, 75, 162, 0.1);
    }
    50% {
        box-shadow:
                0 4px 30px rgba(102, 126, 234, 0.3),
                0 0 60px rgba(118, 75, 162, 0.2);
    }
`;

// Enhanced Header Container with Glass Morphism and Dynamic Background
const HeaderContainer = styled.header`
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: linear-gradient(135deg,
    #667eea 0%,
    #764ba2 25%,
    #f093fb 50%,
    #764ba2 75%,
    #667eea 100%
    );
    background-size: 400% 400%;
    animation: ${gradientShift} 20s ease infinite;
    backdrop-filter: blur(25px) saturate(200%);
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1rem;
    box-shadow:
            0 4px 32px rgba(102, 126, 234, 0.25),
            0 2px 16px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
    animation: ${floatingGlow} 8s ease-in-out infinite;

    /* Enhanced layered background effects */
    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
                radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 80% 70%, rgba(240, 147, 251, 0.1) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(102, 126, 234, 0.08) 0%, transparent 60%);
        pointer-events: none;
        z-index: 1;
    }

    /* Subtle pattern overlay */
    &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.02'%3E%3Cpath d='M20 20c0-5.5-4.5-10-10-10s-10 4.5-10 10 4.5 10 10 10 10-4.5 10-10zm10 0c0-5.5-4.5-10-10-10s-10 4.5-10 10 4.5 10 10 10 10-4.5 10-10z'/%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 1;
        opacity: 0.3;
    }

    /* Entrance animation */
    animation: ${slideDown} 0.5s cubic-bezier(0.4, 0, 0.2, 1), ${floatingGlow} 8s ease-in-out infinite;

    /* Top highlight */
    border-top: 2px solid rgba(255, 255, 255, 0.2);

    @media (min-width: 769px) {
        display: none;
    }
`;

// Enhanced Left Section with Better Spacing
const LeftSection = styled.div`
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    z-index: 10;

    /* Subtle glow effect */
    &::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.05) 0%, transparent 70%);
        transform: translate(-50%, -50%);
        pointer-events: none;
        animation: ${breathe} 4s ease-in-out infinite;
    }
`;

// Enhanced Menu Button with Advanced Interactions
const MenuButton = styled.button`
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: white;
    width: 46px;
    height: 46px;
    border-radius: 14px;
    cursor: pointer;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(15px);
    box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);

    /* Ripple effect background */
    &::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.3) 0%, transparent 70%);
        border-radius: 50%;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        transform: translate(-50%, -50%);
        z-index: 1;
    }

    /* Shimmer effect */
    &::after {
        content: '';
        position: absolute;
        top: 0;
        left: -150px;
        width: 100px;
        height: 100%;
        background: linear-gradient(90deg,
        transparent 0%,
        rgba(255, 255, 255, 0.3) 50%,
        transparent 100%
        );
        transform: skewX(-20deg);
        transition: left 0.6s ease;
        z-index: 2;
    }

    &:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
        transform: translateY(-2px) scale(1.02);
        box-shadow:
                0 8px 25px rgba(0, 0, 0, 0.15),
                0 0 20px rgba(255, 255, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);

        &::before {
            width: 100px;
            height: 100px;
        }

        &::after {
            left: 150px;
        }
    }

    &:active {
        transform: translateY(0) scale(0.98);
        transition: all 0.1s ease;

        &::before {
            animation: ${rippleEffect} 0.6s ease-out;
        }
    }

    /* Icon styling */
    i {
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        position: relative;
        z-index: 3;
        transition: all 0.3s ease;
    }

    &:hover i {
        transform: scale(1.1);
        filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.4));
    }
`;

// Enhanced Header Title with Typography Improvements
const HeaderTitle = styled.div`
    display: flex;
    align-items: center;
    gap: 0.75rem;
    position: relative;
    z-index: 10;

    .logo {
        font-size: 1.6rem;
        animation: ${bounce} 3s ease-in-out infinite;
        filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.3));
        transition: all 0.3s ease;
        cursor: pointer;

        &:hover {
            transform: scale(1.1) rotate(5deg);
            filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.4));
        }
    }

    .title {
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg,
        #ffffff 0%,
        rgba(255, 255, 255, 0.9) 50%,
        rgba(240, 147, 251, 0.8) 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        letter-spacing: -0.02em;
        transition: all 0.3s ease;

        &::after {
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 60%;
            height: 2px;
            background: linear-gradient(90deg,
            rgba(255, 255, 255, 0.6) 0%,
            rgba(240, 147, 251, 0.4) 50%,
            transparent 100%
            );
            border-radius: 1px;
        }

        &:hover {
            transform: translateY(-1px);
            text-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
    }
`;

// Enhanced Right Section
const RightSection = styled.div`
    display: flex;
    align-items: center;
    gap: 0.75rem;
    position: relative;
    z-index: 10;
`;

// Enhanced Status Indicator with Modern Design
const StatusIndicator = styled.div`
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(15px);
    transition: all 0.3s ease;
    cursor: pointer;
    box-shadow: 
        0 2px 8px rgba(0, 0, 0, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);

    &:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
        transform: translateY(-1px);
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }

    .status-dot {
        width: 10px;
        height: 10px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 50%;
        box-shadow: 
            0 0 12px rgba(16, 185, 129, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        animation: ${pulse} 2.5s ease-in-out infinite;
        position: relative;

        &::after {
            content: '';
            position: absolute;
            top: -3px;
            left: -3px;
            right: -3px;
            bottom: -3px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.2) 0%, transparent 70%);
            border-radius: 50%;
            animation: ${pulse} 2.5s ease-in-out infinite reverse;
        }
    }

    .status-text {
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.8rem;
        font-weight: 600;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        letter-spacing: 0.02em;
    }
`;

// Enhanced Action Button with Better Interactions
const ActionButton = styled.button`
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: white;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    backdrop-filter: blur(15px);
    position: relative;
    overflow: hidden;
    box-shadow: 
        0 2px 8px rgba(0, 0, 0, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.15);

    /* Shimmer effect */
    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100px;
        width: 80px;
        height: 100%;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.25) 50%,
            transparent 100%
        );
        transform: skewX(-20deg);
        transition: left 0.6s ease;
    }

    &:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
        transform: translateY(-2px) scale(1.05);
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.15),
            0 0 15px rgba(255, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.25);

        &::before {
            left: 100px;
        }
    }

    &:active {
        transform: translateY(0) scale(0.95);
        transition: all 0.1s ease;
    }

    i {
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        position: relative;
        z-index: 1;
        transition: all 0.3s ease;
    }

    &:hover i {
        transform: scale(1.1) rotate(90deg);
        filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.4));
    }
`;

// Enhanced Mobile Header Component with State Management
const MobileHeader = ({ onMenuClick, title = "Ceylon Guide" }) => {
    const [isOnline, setIsOnline] = useState(navigator.onLine);
    const [lastInteraction, setLastInteraction] = useState(Date.now());

    // Monitor online/offline status
    useEffect(() => {
        const handleOnline = () => setIsOnline(true);
        const handleOffline = () => setIsOnline(false);

        window.addEventListener('online', handleOnline);
        window.addEventListener('offline', handleOffline);

        return () => {
            window.removeEventListener('online', handleOnline);
            window.removeEventListener('offline', handleOffline);
        };
    }, []);

    // Enhanced menu click handler with feedback
    const handleMenuClick = () => {
        setLastInteraction(Date.now());
        // YOUR ORIGINAL FUNCTIONALITY PRESERVED
        onMenuClick();
    };

    // Enhanced settings click handler
    const handleSettingsClick = () => {
        setLastInteraction(Date.now());
        // Add haptic feedback if available
        if (navigator.vibrate) {
            navigator.vibrate(50);
        }
        // Placeholder for settings functionality
        console.log('Settings clicked');
    };

    return (
        <HeaderContainer>
            <LeftSection>
                {/* YOUR ORIGINAL MENU BUTTON - FUNCTIONALITY PRESERVED */}
                <MenuButton
                    onClick={handleMenuClick}
                    aria-label="Open navigation menu"
                    role="button"
                >
                    <i className="fas fa-bars"></i>
                </MenuButton>

                {/* YOUR ORIGINAL HEADER TITLE - ENHANCED STYLING */}
                <HeaderTitle>
                    <div className="logo" role="img" aria-label="Sri Lankan flag">🇱🇰</div>
                    <div className="title">{title}</div>
                </HeaderTitle>
            </LeftSection>

            <RightSection>
                {/* ENHANCED STATUS INDICATOR WITH REAL STATUS */}
                <StatusIndicator
                    role="status"
                    aria-label={`Connection status: ${isOnline ? 'Online' : 'Offline'}`}
                    title={`Last updated: ${new Date(lastInteraction).toLocaleTimeString()}`}
                >
                    <div
                        className="status-dot"
                        style={{
                            background: isOnline
                                ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
                                : 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                            boxShadow: isOnline
                                ? '0 0 12px rgba(16, 185, 129, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
                                : '0 0 12px rgba(239, 68, 68, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
                        }}
                    ></div>
                    <div className="status-text">
                        {isOnline ? 'Online' : 'Offline'}
                    </div>
                </StatusIndicator>

                {/* ENHANCED SETTINGS BUTTON */}
                <ActionButton
                    onClick={handleSettingsClick}
                    aria-label="Open settings"
                    role="button"
                >
                    <i className="fas fa-cog"></i>
                </ActionButton>
            </RightSection>
        </HeaderContainer>
    );
};

export default MobileHeader;