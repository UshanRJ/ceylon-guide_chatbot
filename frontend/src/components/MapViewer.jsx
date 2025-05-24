// frontend/src/components/MapViewer.jsx - ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useEffect, useRef, useState } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { FiMapPin, FiExternalLink, FiRefreshCw, FiMaximize2, FiX, FiNavigation, FiEye } from 'react-icons/fi';

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

const pulse = keyframes`
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
    }
    50% {
        transform: scale(1.02);
        box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
    }
`;

const spin = keyframes`
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
`;

const shimmer = keyframes`
    0% {
        background-position: -200px 0;
    }
    100% {
        background-position: calc(200px + 100%) 0;
    }
`;

const breathe = keyframes`
    0%, 100% {
        transform: scale(1);
        opacity: 0.8;
    }
    50% {
        transform: scale(1.05);
        opacity: 1;
    }
`;

const ripple = keyframes`
    0% {
        transform: scale(0);
        opacity: 1;
    }
    100% {
        transform: scale(4);
        opacity: 0;
    }
`;

const bounceIn = keyframes`
    0% {
        opacity: 0;
        transform: scale(0.3) translate(-50%, -50%);
    }
    50% {
        opacity: 1;
        transform: scale(1.05) translate(-50%, -50%);
    }
    70% {
        transform: scale(0.9) translate(-50%, -50%);
    }
    100% {
        opacity: 1;
        transform: scale(1) translate(-50%, -50%);
    }
`;

const float = keyframes`
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-5px);
    }
`;

// Enhanced Map Container with Glass Morphism and Stunning Visuals
const MapContainer = styled.div`
    width: 100%;
    height: 100%;
    border-radius: 16px;
    overflow: hidden;
    position: relative;
    background: linear-gradient(135deg,
    rgba(102, 126, 234, 0.1) 0%,
    rgba(118, 75, 162, 0.1) 50%,
    rgba(240, 147, 251, 0.1) 100%
    );
    backdrop-filter: blur(20px);
    border: 2px solid;
    border-image: linear-gradient(135deg,
    rgba(102, 126, 234, 0.3) 0%,
    rgba(118, 75, 162, 0.3) 50%,
    rgba(240, 147, 251, 0.3) 100%
    ) 1;
    box-shadow:
            0 20px 60px rgba(102, 126, 234, 0.15),
            0 8px 32px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    animation: ${fadeIn} 0.6s ease-out;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
                radial-gradient(circle at 25% 25%, rgba(102, 126, 234, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, rgba(240, 147, 251, 0.08) 0%, transparent 50%),
                linear-gradient(135deg,
                rgba(255, 255, 255, 0.05) 0%,
                transparent 50%,
                rgba(102, 126, 234, 0.05) 100%
                );
        pointer-events: none;
        z-index: 1;
        border-radius: inherit;
    }

    /* Animated border glow */
    &::after {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg,
        #667eea, #764ba2, #f093fb, #667eea
        );
        background-size: 400% 400%;
        border-radius: 18px;
        z-index: -1;
        animation: ${gradientShift} 8s ease infinite;
        opacity: 0.6;
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

// Enhanced Map Frame
const MapFrame = styled.iframe`
    width: 100%;
    height: calc(100% - 60px);
    margin-top: 60px;
    border: none;
    border-radius: 0 0 16px 16px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    opacity: ${props => props.loaded ? 1 : 0};
    transform: ${props => props.loaded ? 'scale(1)' : 'scale(1.02)'};
    position: relative;
    z-index: 2;
`;

// Enhanced Map Placeholder with Modern Design
const MapPlaceholder = styled.div`
    position: absolute;
    top: 60px;
    left: 0;
    width: 100%;
    height: calc(100% - 60px);
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
            135deg,
            rgba(102, 126, 234, 0.95) 0%,
            rgba(118, 75, 162, 0.95) 30%,
            rgba(240, 147, 251, 0.95) 60%,
            rgba(102, 126, 234, 0.95) 100%
    );
    background-size: 400% 400%;
    animation: ${gradientShift} 15s ease infinite;
    color: white;
    border-radius: 0 0 16px 16px;
    flex-direction: column;
    gap: 1.5rem;
    z-index: ${props => props.show ? 10 : -1};
    opacity: ${props => props.show ? 1 : 0};
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(20px);

    /* Enhanced overlay patterns */
    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
                radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 40%),
                radial-gradient(circle at 40% 60%, rgba(255, 255, 255, 0.08) 0%, transparent 40%);
        pointer-events: none;
        border-radius: inherit;
    }

    /* Animated geometric pattern */
    &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M30 0l30 30-30 30L0 30z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
        opacity: 0.5;
        animation: ${float} 8s ease-in-out infinite;
    }
`;

// Enhanced Loading Spinner
const LoadingSpinner = styled.div`
    width: 50px;
    height: 50px;
    border: 4px solid rgba(255, 255, 255, 0.2);
    border-top: 4px solid white;
    border-radius: 50%;
    animation: ${spin} 1.2s linear infinite;
    position: relative;
    z-index: 1;

    &::after {
        content: '';
        position: absolute;
        top: -8px;
        left: -8px;
        right: -8px;
        bottom: -8px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        animation: ${spin} 2s linear infinite reverse;
    }
`;

// Enhanced Loading Text
const LoadingText = styled.div`
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    position: relative;
    z-index: 1;

    &::after {
        content: '';
        display: inline-block;
        width: 0;
        animation: dots 1.5s steps(4, end) infinite;
    }

    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
`;

// Enhanced Error Container
const ErrorContainer = styled.div`
    text-align: center;
    padding: 2rem;
    position: relative;
    z-index: 1;
    animation: ${slideUp} 0.5s ease-out;
`;

// Enhanced Error Message
const ErrorMessage = styled.div`
    color: #fff;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    font-size: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);

    svg {
        margin-bottom: 0.75rem;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
    }
`;

// Enhanced Map Controls
const MapControls = styled.div`
    position: absolute;
    top: 15px;
    right: 15px;
    display: flex;
    gap: 0.75rem;
    z-index: 20;
    animation: ${fadeIn} 0.8s ease-out 0.3s both;
`;

// Tool Header with Close Button
const ToolHeader = styled.div`
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: linear-gradient(135deg,
    rgba(102, 126, 234, 0.95) 0%,
    rgba(118, 75, 162, 0.95) 50%,
    rgba(240, 147, 251, 0.95) 100%
    );
    backdrop-filter: blur(25px);
    border-radius: 16px 16px 0 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem;
    z-index: 25;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
                radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
        border-radius: inherit;
    }
`;

const ToolTitle = styled.h2`
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    position: relative;
    z-index: 1;

    svg {
        font-size: 1.4rem;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        animation: ${pulse} 2s ease-in-out infinite;
    }
`;

const CloseToolButton = styled.button`
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 8px;
    color: white;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 1.1rem;
    position: relative;
    z-index: 1;
    
    &:hover {
        background: rgba(239, 68, 68, 0.8);
        border-color: rgba(239, 68, 68, 0.5);
        transform: scale(1.1) rotate(90deg);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    &:active {
        transform: scale(0.95) rotate(90deg);
    }
`;

// Enhanced Control Button
const ControlButton = styled.button`
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 10px;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color: #374151;
    box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.1),
            0 2px 8px rgba(0, 0, 0, 0.05);
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
        background: white;
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px) scale(1.05);
        box-shadow:
                0 8px 25px rgba(0, 0, 0, 0.15),
                0 4px 12px rgba(102, 126, 234, 0.2);
        color: #667eea;

        &::before {
            left: 100%;
        }
    }

    &:active {
        transform: translateY(0) scale(0.98);
        transition: all 0.1s ease;
    }

    svg {
        transition: all 0.3s ease;
        filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
    }

    &:hover svg {
        transform: scale(1.1);
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
    }
`;

// Enhanced Location Info
const LocationInfo = styled.div`
    position: absolute;
    bottom: 15px;
    left: 15px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    padding: 1rem 1.25rem;
    border-radius: 12px;
    color: #374151;
    font-size: 0.95rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.1),
            0 4px 16px rgba(102, 126, 234, 0.1);
    max-width: calc(100% - 30px);
    z-index: 20;
    border: 1px solid rgba(255, 255, 255, 0.3);
    animation: ${slideUp} 0.6s ease-out 0.5s both;

    svg {
        color: #667eea;
        filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
        flex-shrink: 0;
    }

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg,
        rgba(102, 126, 234, 0.05) 0%,
        transparent 50%
        );
        border-radius: inherit;
        pointer-events: none;
    }
`;

// Enhanced Coordinate Text
const CoordinateText = styled.div`
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.25rem;
    font-weight: 400;
    font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
`;

// Enhanced Action Button
const ActionButton = styled.button`
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.25rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem auto;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
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
        rgba(255, 255, 255, 0.2) 50%,
        transparent 100%
        );
        transition: left 0.6s ease;
    }

    &:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);

        &::before {
            left: 100%;
        }
    }

    &:active {
        transform: translateY(-1px);
        transition: all 0.1s ease;
    }

    svg {
        transition: all 0.3s ease;
    }

    &:hover svg {
        transform: scale(1.1);
    }
`;

// Enhanced Retry Button
const RetryButton = styled(ActionButton)`
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);

    &:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }
`;

// Enhanced Fullscreen Modal
const FullscreenModal = styled.div`
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.9);
    backdrop-filter: blur(8px);
    z-index: 1000;
    display: ${props => props.show ? 'flex' : 'none'};
    align-items: center;
    justify-content: center;
    padding: 2rem;
    opacity: ${props => props.show ? 1 : 0};
    transition: all 0.3s ease;
    animation: ${props => props.show ? fadeIn : 'none'} 0.3s ease-out;
`;

// Enhanced Fullscreen Map
const FullscreenMap = styled.div`
    width: 100%;
    height: 100%;
    max-width: 1400px;
    max-height: 900px;
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 25px 100px rgba(0, 0, 0, 0.5);
    animation: ${bounceIn} 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    transform-origin: center center;

    &::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        border-radius: 22px;
        z-index: -1;
        animation: ${pulse} 3s ease-in-out infinite;
    }
`;

// Enhanced Close Button
const CloseButton = styled.button`
    position: absolute;
    top: 20px;
    right: 20px;
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(10px);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1.5rem;
    z-index: 1001;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    
    &:hover {
        background: rgba(239, 68, 68, 0.9);
        border-color: rgba(239, 68, 68, 0.5);
        transform: scale(1.1) rotate(90deg);
        box-shadow: 0 8px 30px rgba(239, 68, 68, 0.4);
    }
    
    &:active {
        transform: scale(0.95) rotate(90deg);
    }
`;

// Retry Counter Display
const RetryCounter = styled.div`
    font-size: 0.85rem;
    opacity: 0.9;
    background: rgba(255, 255, 255, 0.1);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: ${pulse} 2s ease-in-out infinite;
`;

// Enhanced MapViewer Component
const MapViewer = ({
                       lat,
                       lng,
                       locationName,
                       zoom = 15,
                       showControls = true,
                       showLocationInfo = true,
                       mapType = 'roadmap',
                       onClose // NEW PROP for closing the tool
                   }) => {
    // YOUR ORIGINAL STATE - COMPLETELY PRESERVED
    const [mapLoaded, setMapLoaded] = useState(false);
    const [mapError, setMapError] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const iframeRef = useRef(null);
    const timeoutRef = useRef(null);

    // YOUR ORIGINAL useEffect - COMPLETELY PRESERVED
    useEffect(() => {
        // Reset states when props change
        setMapLoaded(false);
        setMapError(false);
        setRetryCount(0);

        // Set a timeout to show error if map doesn't load
        timeoutRef.current = setTimeout(() => {
            if (!mapLoaded) {
                setMapError(true);
            }
        }, 10000); // 10 second timeout

        return () => {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, [lat, lng, mapLoaded]);

    // YOUR ORIGINAL HANDLERS - COMPLETELY PRESERVED
    const handleMapLoad = () => {
        setMapLoaded(true);
        setMapError(false);
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }
    };

    const handleMapError = () => {
        setMapError(true);
        setMapLoaded(false);
    };

    const handleRetry = () => {
        setMapLoaded(false);
        setMapError(false);
        setRetryCount(prev => prev + 1);

        // Force iframe reload
        if (iframeRef.current) {
            const currentSrc = iframeRef.current.src;
            iframeRef.current.src = '';
            setTimeout(() => {
                iframeRef.current.src = currentSrc;
            }, 100);
        }

        // Add haptic feedback if available
        if (navigator.vibrate) {
            navigator.vibrate(50);
        }
    };

    const handleFullscreen = () => {
        setIsFullscreen(true);
        // Add haptic feedback if available
        if (navigator.vibrate) {
            navigator.vibrate([50, 100, 50]);
        }
    };

    const handleCloseFullscreen = () => {
        setIsFullscreen(false);
    };

    // YOUR ORIGINAL URL GENERATORS - COMPLETELY PRESERVED
    const getMapUrl = (isFullscreenView = false) => {
        const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

        if (apiKey) {
            // Use Google Maps Embed API
            const baseUrl = 'https://www.google.com/maps/embed/v1/place';
            const params = new URLSearchParams({
                key: apiKey,
                q: `${lat},${lng}`,
                zoom: isFullscreenView ? zoom + 1 : zoom,
                maptype: mapType
            });
            return `${baseUrl}?${params.toString()}`;
        } else {
            // Fallback to OpenStreetMap
            const bbox = 0.01;
            const params = new URLSearchParams({
                bbox: `${lng-bbox},${lat-bbox},${lng+bbox},${lat+bbox}`,
                layer: 'mapnik',
                marker: `${lat},${lng}`
            });
            return `https://www.openstreetmap.org/export/embed.html?${params.toString()}`;
        }
    };

    const getExternalMapUrl = () => {
        return `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
    };

    const getDirectionsUrl = () => {
        return `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`;
    };

    // YOUR ORIGINAL COORDINATE FORMATTER - COMPLETELY PRESERVED
    const formatCoordinate = (coord, type) => {
        const abs = Math.abs(coord);
        const degrees = Math.floor(abs);
        const minutes = Math.floor((abs - degrees) * 60);
        const seconds = Math.floor(((abs - degrees) * 60 - minutes) * 60);
        const direction = type === 'lat' ? (coord >= 0 ? 'N' : 'S') : (coord >= 0 ? 'E' : 'W');
        return `${degrees}°${minutes}'${seconds}"${direction}`;
    };

    // YOUR ORIGINAL VALIDATION - ENHANCED STYLING
    if (!lat || !lng || isNaN(lat) || isNaN(lng)) {
        return (
            <MapContainer>
                <ToolHeader>
                    <ToolTitle>
                        <FiMapPin />
                        Interactive Map
                    </ToolTitle>
                    {onClose && (
                        <CloseToolButton
                            onClick={onClose}
                            aria-label="Close map tool"
                            title="Close map"
                        >
                            <FiX />
                        </CloseToolButton>
                    )}
                </ToolHeader>
                <MapPlaceholder show={true}>
                    <ErrorContainer>
                        <ErrorMessage>
                            <FiMapPin size={32} style={{ marginBottom: '1rem' }} />
                            <div style={{ fontSize: '1.1rem', fontWeight: '600' }}>
                                Invalid coordinates provided
                            </div>
                            <div style={{ fontSize: '0.9rem', marginTop: '0.75rem', opacity: 0.9 }}>
                                Please check the location data and try again
                            </div>
                        </ErrorMessage>
                    </ErrorContainer>
                </MapPlaceholder>
            </MapContainer>
        );
    }

    return (
        <>
            <MapContainer>
                {/* NEW: Tool Header with Close Button */}
                <ToolHeader>
                    <ToolTitle>
                        <FiMapPin />
                        {locationName || 'Interactive Map'}
                    </ToolTitle>
                    {onClose && (
                        <CloseToolButton
                            onClick={onClose}
                            aria-label="Close map tool"
                            title="Close map"
                        >
                            <FiX />
                        </CloseToolButton>
                    )}
                </ToolHeader>
                {/* YOUR ORIGINAL MAP CONTROLS - ENHANCED STYLING */}
                {showControls && (
                    <MapControls>
                        <ControlButton
                            onClick={handleRetry}
                            title="Refresh Map"
                            aria-label="Refresh map"
                        >
                            <FiRefreshCw />
                        </ControlButton>
                        <ControlButton
                            onClick={handleFullscreen}
                            title="Fullscreen View"
                            aria-label="Open fullscreen view"
                        >
                            <FiMaximize2 />
                        </ControlButton>
                    </MapControls>
                )}

                {/* YOUR ORIGINAL LOCATION INFO - ENHANCED STYLING */}
                {showLocationInfo && locationName && (
                    <LocationInfo>
                        <FiMapPin />
                        <div>
                            <div style={{ fontWeight: '600' }}>{locationName}</div>
                            <CoordinateText>
                                {formatCoordinate(lat, 'lat')}, {formatCoordinate(lng, 'lng')}
                            </CoordinateText>
                        </div>
                    </LocationInfo>
                )}

                {/* YOUR ORIGINAL LOADING/ERROR PLACEHOLDER - ENHANCED STYLING */}
                <MapPlaceholder show={!mapLoaded || mapError}>
                    {!mapError ? (
                        <>
                            <LoadingSpinner />
                            <LoadingText>Loading interactive map</LoadingText>
                            {retryCount > 0 && (
                                <RetryCounter>
                                    Retry attempt: {retryCount}
                                </RetryCounter>
                            )}
                        </>
                    ) : (
                        <ErrorContainer>
                            <ErrorMessage>
                                <FiMapPin size={32} style={{ marginBottom: '1rem' }} />
                                <div style={{ fontSize: '1.1rem', fontWeight: '600' }}>
                                    Unable to load map
                                </div>
                                <div style={{ fontSize: '0.9rem', marginTop: '0.75rem', opacity: 0.9 }}>
                                    Please check your internet connection and try again
                                </div>
                            </ErrorMessage>

                            <RetryButton onClick={handleRetry}>
                                <FiRefreshCw />
                                Try Again
                            </RetryButton>

                            <ActionButton
                                onClick={() => window.open(getExternalMapUrl(), '_blank')}
                            >
                                <FiEye />
                                View on Google Maps
                            </ActionButton>

                            <ActionButton
                                onClick={() => window.open(getDirectionsUrl(), '_blank')}
                            >
                                <FiNavigation />
                                Get Directions
                            </ActionButton>
                        </ErrorContainer>
                    )}
                </MapPlaceholder>

                {/* YOUR ORIGINAL MAP FRAME - ENHANCED STYLING */}
                <MapFrame
                    ref={iframeRef}
                    src={getMapUrl()}
                    onLoad={handleMapLoad}
                    onError={handleMapError}
                    title={`Interactive map of ${locationName || 'Location'}`}
                    loaded={mapLoaded && !mapError}
                    allow="geolocation"
                />
            </MapContainer>

            {/* YOUR ORIGINAL FULLSCREEN MODAL - ENHANCED STYLING */}
            <FullscreenModal show={isFullscreen}>
                <CloseButton
                    onClick={handleCloseFullscreen}
                    aria-label="Close fullscreen view"
                >
                    <FiX />
                </CloseButton>
                <FullscreenMap>
                    <MapViewer
                        lat={lat}
                        lng={lng}
                        locationName={locationName}
                        zoom={zoom + 1}
                        showControls={false}
                        showLocationInfo={true}
                        mapType={mapType}
                    />
                </FullscreenMap>
            </FullscreenModal>
        </>
    );
};

export default MapViewer;