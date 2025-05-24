// frontend/src/App.js - ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState, useEffect } from 'react';
import styled, { createGlobalStyle, ThemeProvider, keyframes } from 'styled-components';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import MobileHeader from './components/MobileHeader';

// Enhanced Animations
const fadeIn = keyframes`
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
`;

const slideIn = keyframes`
    from {
        transform: translateX(-100%);
    }
    to {
        transform: translateX(0);
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

// Enhanced Global Styles with Modern Design System
const GlobalStyle = createGlobalStyle`
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        overflow: hidden;
        line-height: 1.6;
        color: #2d3748;
    }

    /* Enhanced Custom Scrollbar with Modern Design */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #5a67d8, #6b46c1);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    /* Selection styling */
    ::selection {
        background: rgba(102, 126, 234, 0.2);
        color: inherit;
    }

    /* Focus styles for accessibility */
    *:focus-visible {
        outline: 2px solid #667eea;
        outline-offset: 2px;
        border-radius: 4px;
    }
`;

// Enhanced Theme with Modern Color Palette and Design Tokens
const theme = {
    colors: {
        primary: '#667eea',
        primaryDark: '#5a67d8',
        secondary: '#764ba2',
        secondaryDark: '#6b46c1',
        accent: '#f093fb',
        success: '#48bb78',
        warning: '#ed8936',
        danger: '#f56565',
        info: '#4299e1',
        dark: '#1a202c',
        light: '#ffffff',
        background: {
            primary: '#ffffff',
            secondary: '#f7fafc',
            tertiary: '#edf2f7',
            overlay: 'rgba(0, 0, 0, 0.6)'
        },
        text: {
            primary: '#2d3748',
            secondary: '#4a5568',
            tertiary: '#718096',
            inverse: '#ffffff'
        },
        border: {
            light: '#e2e8f0',
            medium: '#cbd5e0',
            dark: '#a0aec0'
        },
        gray: {
            50: '#f7fafc',
            100: '#edf2f7',
            200: '#e2e8f0',
            300: '#cbd5e0',
            400: '#a0aec0',
            500: '#718096',
            600: '#4a5568',
            700: '#2d3748',
            800: '#1a202c',
            900: '#171923'
        }
    },
    shadows: {
        sm: '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
        md: '0 4px 6px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.06)',
        lg: '0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05)',
        xl: '0 20px 25px rgba(0, 0, 0, 0.15), 0 10px 10px rgba(0, 0, 0, 0.04)',
        floating: '0 25px 50px rgba(0, 0, 0, 0.15)',
        glow: '0 0 20px rgba(102, 126, 234, 0.3)'
    },
    breakpoints: {
        mobile: '640px',
        tablet: '768px',
        desktop: '1024px',
        wide: '1280px'
    },
    spacing: {
        xs: '0.25rem',
        sm: '0.5rem',
        md: '1rem',
        lg: '1.5rem',
        xl: '2rem',
        xxl: '3rem'
    },
    borderRadius: {
        sm: '4px',
        md: '8px',
        lg: '12px',
        xl: '16px',
        full: '9999px'
    }
};

// Enhanced App Container with Glass Morphism Effect
const AppContainer = styled.div`
    display: flex;
    height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
    position: relative;
    overflow: hidden;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(102, 126, 234, 0.2) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
`;

// Enhanced Main Content with Glass Effect
const MainContent = styled.main`
    flex: 1;
    display: flex;
    flex-direction: column;
    position: relative;
    margin-left: ${props => props.$sidebarOpen ? '280px' : '0'};
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(10px);
    border-left: 1px solid rgba(255, 255, 255, 0.1);
    z-index: 1;
    animation: ${fadeIn} 0.6s ease-out;

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        margin-left: 0;
        border-left: none;
        background: rgba(255, 255, 255, 0.05);
    }

    /* Add subtle inner shadow for depth */
    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(255, 255, 255, 0.2) 50%, 
            transparent 100%);
        pointer-events: none;
    }
`;

// Enhanced Mobile Overlay with Improved Animation
const MobileOverlay = styled.div`
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: ${props => props.theme.colors.background.overlay};
    backdrop-filter: blur(4px);
    z-index: 998;
    opacity: ${props => props.$show ? 1 : 0};
    pointer-events: ${props => props.$show ? 'all' : 'none'};
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

    @media (max-width: ${props => props.theme.breakpoints.tablet}) {
        display: block;
    }
`;

// Loading Component for Better UX
const LoadingContainer = styled.div`
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    opacity: ${props => props.$show ? 1 : 0};
    pointer-events: ${props => props.$show ? 'all' : 'none'};
    transition: opacity 0.5s ease;
`;

const LoadingSpinner = styled.div`
    width: 50px;
    height: 50px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-top: 3px solid white;
    border-radius: 50%;
    animation: spin 1s linear infinite;

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;

// Enhanced App Component with Improved UX
function App() {
    // YOUR ORIGINAL STATE - COMPLETELY PRESERVED
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
    const [activeView, setActiveView] = useState('chat');
    const [isMobile, setIsMobile] = useState(false);

    // Enhanced state for better UX
    const [isLoading, setIsLoading] = useState(true);
    const [isInitialized, setIsInitialized] = useState(false);

    // Enhanced initialization effect
    useEffect(() => {
        const initializeApp = async () => {
            // Simulate app initialization
            await new Promise(resolve => setTimeout(resolve, 1000));
            setIsLoading(false);

            // Trigger entrance animation
            setTimeout(() => setIsInitialized(true), 100);
        };

        initializeApp();
    }, []);

    // YOUR ORIGINAL useEffect - ENHANCED WITH PERFORMANCE OPTIMIZATION
    useEffect(() => {
        const checkMobile = () => {
            const mobile = window.innerWidth <= 768;
            setIsMobile(mobile);
            if (mobile) {
                setSidebarOpen(false);
            } else {
                setSidebarOpen(true);
                setMobileSidebarOpen(false);
            }
        };

        // Debounce resize for better performance
        let timeoutId;
        const debouncedCheckMobile = () => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(checkMobile, 150);
        };

        checkMobile();
        window.addEventListener('resize', debouncedCheckMobile);
        return () => {
            window.removeEventListener('resize', debouncedCheckMobile);
            clearTimeout(timeoutId);
        };
    }, []);

    // YOUR ORIGINAL HANDLERS - ENHANCED WITH BETTER UX
    const handleViewChange = (view) => {
        setActiveView(view);
        if (isMobile && mobileSidebarOpen) {
            // Add slight delay for smoother transition
            setTimeout(() => setMobileSidebarOpen(false), 150);
        }
    };

    const toggleMobileSidebar = () => {
        setMobileSidebarOpen(prev => !prev);
    };

    // Enhanced overlay click handler with proper event handling
    const handleOverlayClick = (e) => {
        if (e.target === e.currentTarget) {
            setMobileSidebarOpen(false);
        }
    };

    // YOUR ORIGINAL RETURN STRUCTURE - ENHANCED WITH BETTER UX
    return (
        <ThemeProvider theme={theme}>
            <GlobalStyle />

            {/* Loading Screen */}
            <LoadingContainer $show={isLoading}>
                <LoadingSpinner />
            </LoadingContainer>

            <AppContainer style={{
                opacity: isInitialized ? 1 : 0,
                transform: isInitialized ? 'translateY(0)' : 'translateY(20px)',
                transition: 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)'
            }}>
                {isMobile && (
                    <MobileHeader
                        onMenuClick={toggleMobileSidebar}
                        title="Ceylon Guide"
                    />
                )}

                <Sidebar
                    isOpen={!isMobile ? sidebarOpen : mobileSidebarOpen}
                    onClose={() => isMobile ? setMobileSidebarOpen(false) : setSidebarOpen(false)}
                    activeView={activeView}
                    onViewChange={handleViewChange}
                    isMobile={isMobile}
                />

                <MobileOverlay
                    $show={mobileSidebarOpen}
                    onClick={handleOverlayClick}
                    role="button"
                    tabIndex={-1}
                    aria-label="Close sidebar"
                />

                <MainContent $sidebarOpen={!isMobile && sidebarOpen}>
                    <ChatInterface activeView={activeView} />
                </MainContent>
            </AppContainer>
        </ThemeProvider>
    );
}

export default App;