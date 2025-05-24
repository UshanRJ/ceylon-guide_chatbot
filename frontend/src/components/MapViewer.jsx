// frontend/src/components/MapViewer.jsx
import React, { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { FiMapPin, FiExternalLink, FiRefreshCw, FiMaximize2 } from 'react-icons/fi';

const MapContainer = styled.div`
  width: 100%;
  height: 100%;
  border-radius: 10px;
  overflow: hidden;
  position: relative;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
`;

const MapFrame = styled.iframe`
  width: 100%;
  height: 100%;
  border: none;
  border-radius: 10px;
  transition: opacity 0.3s ease;
  opacity: ${props => props.loaded ? 1 : 0};
`;

const MapPlaceholder = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, rgba(103, 126, 234, 0.8), rgba(118, 75, 162, 0.8));
  color: white;
  border-radius: 10px;
  flex-direction: column;
  gap: 1rem;
  z-index: ${props => props.show ? 2 : -1};
  opacity: ${props => props.show ? 1 : 0};
  transition: all 0.3s ease;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top: 3px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const LoadingText = styled.div`
  font-size: 1rem;
  font-weight: 500;
  text-align: center;
`;

const ErrorContainer = styled.div`
  text-align: center;
  padding: 1rem;
`;

const ErrorMessage = styled.div`
  color: #ff6b6b;
  background: rgba(255, 107, 107, 0.1);
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  font-size: 0.9rem;
`;

const MapControls = styled.div`
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  gap: 0.5rem;
  z-index: 3;
`;

const ControlButton = styled.button`
  background: rgba(255, 255, 255, 0.9);
  border: none;
  border-radius: 6px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #333;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  
  &:hover {
    background: white;
    transform: scale(1.05);
  }
  
  &:active {
    transform: scale(0.95);
  }
`;

const LocationInfo = styled.div`
  position: absolute;
  bottom: 10px;
  left: 10px;
  background: rgba(255, 255, 255, 0.95);
  padding: 0.75rem 1rem;
  border-radius: 8px;
  color: #333;
  font-size: 0.9rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  max-width: calc(100% - 20px);
  z-index: 3;
`;

const CoordinateText = styled.div`
  font-size: 0.8rem;
  color: #666;
  margin-top: 0.25rem;
`;

const ActionButton = styled.button`
  background: #007bff;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  
  &:hover {
    background: #0056b3;
    transform: translateY(-2px);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

const RetryButton = styled(ActionButton)`
  background: #28a745;
  
  &:hover {
    background: #1e7e34;
  }
`;

const FullscreenModal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.9);
  z-index: 1000;
  display: ${props => props.show ? 'flex' : 'none'};
  align-items: center;
  justify-content: center;
  padding: 2rem;
`;

const FullscreenMap = styled.div`
  width: 100%;
  height: 100%;
  max-width: 1200px;
  max-height: 800px;
  position: relative;
  border-radius: 15px;
  overflow: hidden;
`;

const CloseButton = styled.button`
  position: absolute;
  top: 15px;
  right: 15px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 1.2rem;
  z-index: 1001;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(0, 0, 0, 0.9);
    transform: scale(1.1);
  }
`;

const MapViewer = ({
                       lat,
                       lng,
                       locationName,
                       zoom = 15,
                       showControls = true,
                       showLocationInfo = true,
                       mapType = 'roadmap'
                   }) => {
    const [mapLoaded, setMapLoaded] = useState(false);
    const [mapError, setMapError] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const iframeRef = useRef(null);
    const timeoutRef = useRef(null);

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
    };

    const handleFullscreen = () => {
        setIsFullscreen(true);
    };

    const handleCloseFullscreen = () => {
        setIsFullscreen(false);
    };

    // Generate Google Maps embed URL
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

    // Generate Google Maps link for external viewing
    const getExternalMapUrl = () => {
        return `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
    };

    // Generate Google Maps directions URL
    const getDirectionsUrl = () => {
        return `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`;
    };

    // Format coordinates for display
    const formatCoordinate = (coord, type) => {
        const abs = Math.abs(coord);
        const degrees = Math.floor(abs);
        const minutes = Math.floor((abs - degrees) * 60);
        const seconds = Math.floor(((abs - degrees) * 60 - minutes) * 60);
        const direction = type === 'lat' ? (coord >= 0 ? 'N' : 'S') : (coord >= 0 ? 'E' : 'W');
        return `${degrees}°${minutes}'${seconds}"${direction}`;
    };

    if (!lat || !lng || isNaN(lat) || isNaN(lng)) {
        return (
            <MapContainer>
                <MapPlaceholder show={true}>
                    <ErrorContainer>
                        <ErrorMessage>
                            <FiMapPin size={24} style={{ marginBottom: '0.5rem' }} />
                            <div>Invalid coordinates provided</div>
                            <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', opacity: 0.8 }}>
                                Please check the location data
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
                {/* Map Controls */}
                {showControls && (
                    <MapControls>
                        <ControlButton onClick={handleRetry} title="Refresh Map">
                            <FiRefreshCw />
                        </ControlButton>
                        <ControlButton onClick={handleFullscreen} title="Fullscreen">
                            <FiMaximize2 />
                        </ControlButton>
                    </MapControls>
                )}

                {/* Location Info */}
                {showLocationInfo && locationName && (
                    <LocationInfo>
                        <FiMapPin />
                        <div>
                            <div>{locationName}</div>
                            <CoordinateText>
                                {formatCoordinate(lat, 'lat')}, {formatCoordinate(lng, 'lng')}
                            </CoordinateText>
                        </div>
                    </LocationInfo>
                )}

                {/* Loading/Error Placeholder */}
                <MapPlaceholder show={!mapLoaded || mapError}>
                    {!mapError ? (
                        <>
                            <LoadingSpinner />
                            <LoadingText>Loading map...</LoadingText>
                            {retryCount > 0 && (
                                <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
                                    Retry attempt: {retryCount}
                                </div>
                            )}
                        </>
                    ) : (
                        <ErrorContainer>
                            <ErrorMessage>
                                <FiMapPin size={24} style={{ marginBottom: '0.5rem' }} />
                                <div>Unable to load map</div>
                                <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', opacity: 0.8 }}>
                                    Please check your internet connection
                                </div>
                            </ErrorMessage>

                            <RetryButton onClick={handleRetry}>
                                <FiRefreshCw />
                                Try Again
                            </RetryButton>

                            <ActionButton
                                onClick={() => window.open(getExternalMapUrl(), '_blank')}
                            >
                                <FiExternalLink />
                                View on Google Maps
                            </ActionButton>

                            <ActionButton
                                onClick={() => window.open(getDirectionsUrl(), '_blank')}
                            >
                                <FiMapPin />
                                Get Directions
                            </ActionButton>
                        </ErrorContainer>
                    )}
                </MapPlaceholder>

                {/* Map Frame */}
                <MapFrame
                    ref={iframeRef}
                    src={getMapUrl()}
                    onLoad={handleMapLoad}
                    onError={handleMapError}
                    title={`Map of ${locationName || 'Location'}`}
                    loaded={mapLoaded && !mapError}
                    allow="geolocation"
                />
            </MapContainer>

            {/* Fullscreen Modal */}
            <FullscreenModal show={isFullscreen}>
                <CloseButton onClick={handleCloseFullscreen}>
                    ×
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