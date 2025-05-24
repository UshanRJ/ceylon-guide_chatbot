// frontend/src/components/Sidebar.jsx - ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState, useEffect } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { FiMessageSquare, FiCloud, FiDollarSign, FiGlobe, FiMap, FiX, FiChevronRight } from 'react-icons/fi';

// Enhanced Animations
const slideInFromLeft = keyframes`
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
`;

const fadeInUp = keyframes`
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
`;

const pulse = keyframes`
    0% {
        box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
    }
`;

const shimmerEffect = keyframes`
    0% {
        background-position: -200px 0;
    }
    100% {
        background-position: calc(200px + 100%) 0;
    }
`;

// Enhanced Sidebar Container with Glass Morphism
const SidebarContainer = styled.aside`
    position: fixed;
    left: ${props => props.$isOpen ? '0' : '-300px'};
    top: 0;
    bottom: 0;
    width: 280px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
    z-index: 999;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
    box-shadow: ${props => props.theme.shadows.xl};

    /* Glass effect with subtle gradient */
    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.1) 0%,
                rgba(255, 255, 255, 0.05) 50%,
                rgba(255, 255, 255, 0.1) 100%
        );
        pointer-events: none;
        z-index: -1;
    }

    /* Animated border gradient */
    &::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 2px;
        height: 100%;
        background: linear-gradient(
                180deg,
                transparent 0%,
                rgba(102, 126, 234, 0.5) 20%,
                rgba(118, 75, 162, 0.5) 80%,
                transparent 100%
        );
        opacity: ${props => props.$isOpen ? 1 : 0};
        transition: opacity 0.3s ease;
    }

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        top: 60px;
        box-shadow: ${props => props.theme.shadows.floating};
        border-radius: 0 16px 16px 0;
        width: 300px;
        left: ${props => props.$isOpen ? '0' : '-320px'};
        animation: ${props => props.$isOpen ? slideInFromLeft : 'none'} 0.3s ease-out;
    }
`;

// Enhanced Sidebar Header with Better Visual Hierarchy
const SidebarHeader = styled.div`
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);

    /* Subtle top accent */
    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
`;

// Enhanced Logo with Animation
const Logo = styled.div`
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.25rem;
    font-weight: 700;
    color: ${props => props.theme.colors.text.primary};
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;

    &:hover {
        transform: translateY(-1px);

        span:first-child {
            animation: ${pulse} 1.5s infinite;
        }
    }

    span:first-child {
        font-size: 1.5rem;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
        transition: all 0.3s ease;
    }

    span:last-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
`;

// Enhanced Close Button
const CloseButton = styled.button`
    display: none;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: ${props => props.theme.borderRadius.full};
    color: ${props => props.theme.colors.text.secondary};
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    cursor: pointer;
    transition: all 0.3s ease;

    &:hover {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
        color: #ef4444;
        transform: scale(1.05);
    }

    &:active {
        transform: scale(0.95);
    }

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        display: flex;
    }
`;

// Enhanced Navigation Container
const Navigation = styled.nav`
    flex: 1;
    padding: 1.5rem 0;
    overflow-y: auto;
    overflow-x: hidden;

    /* Custom scrollbar for navigation */
    &::-webkit-scrollbar {
        width: 4px;
    }

    &::-webkit-scrollbar-track {
        background: transparent;
    }

    &::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 2px;
    }
`;

// Enhanced Navigation Item with Modern Design
const NavItem = styled.button`
    width: calc(100% - 1rem);
    margin: 0 0.5rem 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    background: ${props => props.$active
            ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)'
            : 'transparent'
    };
    border: 1px solid ${props => props.$active
            ? 'rgba(102, 126, 234, 0.2)'
            : 'transparent'
    };
    border-radius: ${props => props.theme.borderRadius.lg};
    color: ${props => props.$active
            ? props.theme.colors.primary
            : props.theme.colors.text.secondary
    };
    font-size: 0.95rem;
    font-weight: ${props => props.$active ? '600' : '500'};
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;

    /* Add fadeInUp animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Hover effect with shimmer */
    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: -200px;
        width: 200px;
        height: 100%;
        background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
        );
        transition: left 0.5s ease;
    }

    &:hover {
        background: ${props => props.$active
                ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)'
                : 'rgba(102, 126, 234, 0.05)'
        };
        border-color: rgba(102, 126, 234, 0.2);
        color: ${props => props.theme.colors.primary};
        transform: translateX(4px);
        box-shadow: ${props => props.theme.shadows.md};

        &::before {
            left: calc(100% + 200px);
        }

        .nav-arrow {
            opacity: 1;
            transform: translateX(0);
        }
    }

    &:active {
        transform: translateX(2px) scale(0.98);
    }

    /* Active state indicator */
    ${props => props.$active && css`
        &::after {
            content: '';
            position: absolute;
            left: 0;
            top: 20%;
            bottom: 20%;
            width: 3px;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 0 2px 2px 0;
        }
    `}

    svg {
        font-size: 1.25rem;
        transition: all 0.3s ease;
        flex-shrink: 0;
    }

    .nav-arrow {
        margin-left: auto;
        opacity: 0;
        transform: translateX(-10px);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
`;

// Enhanced Section Container
const NavSection = styled.div`
    margin-bottom: 2rem;

    &:last-child {
        margin-bottom: 1rem;
    }
`;

// Enhanced Section Title
const SectionTitle = styled.h3`
    font-size: 0.75rem;
    font-weight: 700;
    color: ${props => props.theme.colors.text.tertiary};
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 0 1.5rem;
    margin-bottom: 0.75rem;
    position: relative;
    display: flex;
    align-items: center;

    &::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(
                90deg,
                rgba(102, 126, 234, 0.3) 0%,
                transparent 100%
        );
        margin-left: 1rem;
    }
`;

// Enhanced Footer with Gradient Background
const Footer = styled.div`
    padding: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.875rem;
    color: ${props => props.theme.colors.text.tertiary};
    text-align: center;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
    position: relative;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 20%;
        right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(102, 126, 234, 0.3) 50%, transparent 100%);
    }
`;

// Enhanced Sidebar Component with Animation States
const Sidebar = ({ isOpen, onClose, activeView, onViewChange, isMobile }) => {
    const [animatedItems, setAnimatedItems] = useState(new Set());

    // Enhanced menu items with better organization
    const menuItems = [
        {
            section: 'Main',
            items: [
                {
                    id: 'chat',
                    label: 'Chat Assistant',
                    icon: <FiMessageSquare />,
                    description: 'AI-powered conversations'
                }
            ]
        },
        {
            section: 'Tools',
            items: [
                {
                    id: 'weather',
                    label: 'Weather',
                    icon: <FiCloud />,
                    description: 'Current weather conditions'
                },
                {
                    id: 'currency',
                    label: 'Currency Converter',
                    icon: <FiDollarSign />,
                    description: 'Exchange rates & conversion'
                },
                {
                    id: 'translate',
                    label: 'Translator',
                    icon: <FiGlobe />,
                    description: 'Multi-language translation'
                },
                {
                    id: 'map',
                    label: 'Maps',
                    icon: <FiMap />,
                    description: 'Explore and navigate'
                }
            ]
        }
    ];

    // Enhanced item click handler with animation
    const handleItemClick = (itemId) => {
        // Add click animation
        setAnimatedItems(prev => new Set([...prev, itemId]));
        setTimeout(() => {
            setAnimatedItems(prev => {
                const newSet = new Set(prev);
                newSet.delete(itemId);
                return newSet;
            });
        }, 300);

        // YOUR ORIGINAL FUNCTIONALITY PRESERVED
        onViewChange(itemId);
    };

    // Animation effect for menu items when sidebar opens
    useEffect(() => {
        if (isOpen) {
            const timer = setTimeout(() => {
                const items = document.querySelectorAll('.nav-item');
                items.forEach((item, index) => {
                    item.style.animationName = 'fadeInUp';
                    item.style.animationDuration = '0.3s';
                    item.style.animationTimingFunction = 'ease';
                    item.style.animationDelay = `${index * 0.05}s`;
                    item.style.animationFillMode = 'both';
                });
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [isOpen]);

    return (
        <SidebarContainer $isOpen={isOpen}>
            <SidebarHeader>
                <Logo>
                    <span>🇱🇰</span>
                    <span>Ceylon Guide</span>
                </Logo>
                {isMobile && (
                    <CloseButton onClick={onClose} aria-label="Close sidebar">
                        <FiX />
                    </CloseButton>
                )}
            </SidebarHeader>

            <Navigation>
                {menuItems.map((section, sectionIndex) => (
                    <NavSection key={section.section}>
                        <SectionTitle>{section.section}</SectionTitle>
                        {section.items.map((item, itemIndex) => (
                            <NavItem
                                key={item.id}
                                className="nav-item"
                                $active={activeView === item.id}
                                onClick={() => handleItemClick(item.id)}
                                title={item.description}
                                style={{
                                    animationDelay: `${(sectionIndex * section.items.length + itemIndex) * 0.05}s`
                                }}
                            >
                                {item.icon}
                                <span>{item.label}</span>
                                <FiChevronRight className="nav-arrow" />
                            </NavItem>
                        ))}
                    </NavSection>
                ))}
            </Navigation>

            <Footer>
                <div style={{
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                    fontWeight: '600'
                }}>
                    © 2025 Ceylon Guide
                </div>
            </Footer>
        </SidebarContainer>
    );
};

export default Sidebar;