// frontend/src/components/tools/MapWidget.jsx - SIMPLE ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiMapPin, FiSearch, FiNavigation, FiX, FiMap } from 'react-icons/fi';
import chatService from '../../services/chatService';

// Simple fade-in animation
const fadeIn = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
`;

// Clean Container with subtle styling
const Container = styled.div`
  ${fadeIn}
  height: 100%;
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  overflow: hidden;
  animation: fadeIn 0.3s ease-out;
`;

// Simple Header with close button
const Header = styled.div`
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
  padding: 1rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Title = styled.h2`
  font-size: 1.25rem;
  color: #1a202c;
  margin: 0;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  svg {
    color: #4299e1;
  }
`;

// Simple close button
const CloseButton = styled.button`
  background: none;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  color: #64748b;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #fee2e2;
    border-color: #fecaca;
    color: #dc2626;
  }
`;

// Content area with scrolling
const Content = styled.div`
  padding: 1.5rem;
  height: calc(100% - 60px);
  overflow-y: auto;
  
  /* Simple scrollbar styling */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f1f5f9;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
`;

// Clean search form
const SearchForm = styled.form`
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
`;

const SearchInput = styled.input`
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 1rem;
  transition: border-color 0.2s ease;
  
  &::placeholder {
    color: #94a3b8;
  }
  
  &:focus {
    outline: none;
    border-color: #4299e1;
  }
  
  &:disabled {
    background: #f8fafc;
    cursor: not-allowed;
  }
`;

const SearchButton = styled.button`
  padding: 0.75rem 1rem;
  background: #4299e1;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 500;
  transition: background 0.2s ease;
  
  &:hover:not(:disabled) {
    background: #3182ce;
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

// Popular places section
const PopularPlaces = styled.div`
  margin-bottom: 1.5rem;
`;

const SectionLabel = styled.div`
  font-size: 0.875rem;
  color: #64748b;
  margin-bottom: 0.75rem;
  font-weight: 500;
`;

const PlacesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.75rem;
`;

const PlaceButton = styled.button`
  padding: 0.875rem;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: left;
  
  &:hover {
    background: #4299e1;
    color: white;
    border-color: #4299e1;
    transform: translateY(-1px);
  }
  
  .name {
    font-weight: 500;
    font-size: 0.875rem;
    margin-bottom: 0.25rem;
  }
  
  .type {
    font-size: 0.75rem;
    opacity: 0.8;
  }
`;

// Clean map container
const MapContainer = styled.div`
  width: 100%;
  height: 300px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #e2e8f0;
  margin-bottom: 1rem;
  background: #f8fafc;
  animation: fadeIn 0.3s ease-out;
`;

const MapFrame = styled.iframe`
  width: 100%;
  height: 100%;
  border: none;
`;

// Simple location info card
const LocationInfo = styled.div`
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 1rem;
  animation: fadeIn 0.3s ease-out;
`;

const LocationTitle = styled.h3`
  font-size: 1.125rem;
  color: #1a202c;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  
  svg {
    color: #4299e1;
  }
`;

const LocationDetails = styled.div`
  font-size: 0.875rem;
  color: #4a5568;
  line-height: 1.6;
  
  > div {
    margin-bottom: 0.5rem;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
`;

const DirectionsButton = styled.button`
  width: 100%;
  padding: 0.75rem;
  background: #48bb78;
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
  transition: background 0.2s ease;
  
  &:hover {
    background: #38a169;
  }
`;

// Simple error message
const ErrorMessage = styled.div`
  background: #fed7d7;
  color: #c53030;
  padding: 0.875rem;
  border-radius: 6px;
  margin-bottom: 1rem;
  font-size: 0.875rem;
  border: 1px solid #feb2b2;
`;

// Simple loading spinner
const LoadingSpinner = styled.div`
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

// Enhanced MapWidget Component with Simple Design
const MapWidget = ({ onClose }) => { // NEW: Accept onClose prop
    // YOUR ORIGINAL STATE - COMPLETELY PRESERVED
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedLocation, setSelectedLocation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // YOUR ORIGINAL POPULAR PLACES - COMPLETELY PRESERVED
    const popularPlaces = [
        { name: 'Sigiriya', type: 'Ancient Fortress', coords: { lat: 7.957, lng: 80.760 } },
        { name: 'Temple of Tooth', type: 'Sacred Temple', coords: { lat: 7.294, lng: 80.641 } },
        { name: 'Galle Fort', type: 'Historic Fort', coords: { lat: 6.032, lng: 80.215 } },
        { name: "Adam's Peak", type: 'Sacred Mountain', coords: { lat: 6.809, lng: 80.499 } },
        { name: 'Ella Rock', type: 'Scenic Viewpoint', coords: { lat: 6.867, lng: 81.053 } },
        { name: 'Mirissa Beach', type: 'Beach', coords: { lat: 5.945, lng: 80.459 } }
    ];

    // YOUR ORIGINAL SEARCH HANDLER - COMPLETELY PRESERVED
    const handleSearch = async (e) => {
        e.preventDefault();
        if (!searchQuery.trim()) return;

        setLoading(true);
        setError(null);

        try {
            const response = await chatService.getLocationInfo(searchQuery);
            if (response.success) {
                setSelectedLocation({
                    name: response.location,
                    coords: response.coordinates,
                    country: response.country,
                    description: response.description
                });
            } else {
                setError('Location not found. Please try another search.');
            }
        } catch (err) {
            setError('Failed to search location. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    // YOUR ORIGINAL PLACE CLICK HANDLER - COMPLETELY PRESERVED
    const handlePlaceClick = (place) => {
        setSelectedLocation(place);
        setSearchQuery(place.name);
    };

    // YOUR ORIGINAL MAP URL GENERATOR - COMPLETELY PRESERVED
    const getMapUrl = (location) => {
        if (!location || !location.coords) return '';

        const { lat, lng } = location.coords;
        // Using OpenStreetMap as it doesn't require API key
        return `https://www.openstreetmap.org/export/embed.html?bbox=${lng-0.01},${lat-0.01},${lng+0.01},${lat+0.01}&layer=mapnik&marker=${lat},${lng}`;
    };

    // YOUR ORIGINAL DIRECTIONS HANDLER - COMPLETELY PRESERVED
    const openDirections = () => {
        if (selectedLocation && selectedLocation.coords) {
            const { lat, lng } = selectedLocation.coords;
            window.open(`https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`, '_blank');
        }
    };

    return (
        <Container>
            {/* NEW: Simple Header with Close Button */}
            <Header>
                <Title>
                    <FiMap />
                    Maps & Locations
                </Title>
                {onClose && (
                    <CloseButton
                        onClick={onClose}
                        aria-label="Close map widget"
                        title="Close"
                    >
                        <FiX />
                    </CloseButton>
                )}
            </Header>

            <Content>
                {/* YOUR ORIGINAL SEARCH FORM - CLEAN STYLING */}
                <SearchForm onSubmit={handleSearch}>
                    <SearchInput
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search for a location..."
                        disabled={loading}
                    />
                    <SearchButton
                        type="submit"
                        disabled={loading || !searchQuery.trim()}
                    >
                        {loading ? <LoadingSpinner /> : <FiSearch />}
                        Search
                    </SearchButton>
                </SearchForm>

                {/* YOUR ORIGINAL POPULAR PLACES - CLEAN STYLING */}
                <PopularPlaces>
                    <SectionLabel>
                        Popular destinations:
                    </SectionLabel>
                    <PlacesGrid>
                        {popularPlaces.map((place, index) => (
                            <PlaceButton
                                key={index}
                                onClick={() => handlePlaceClick(place)}
                            >
                                <div className="name">{place.name}</div>
                                <div className="type">{place.type}</div>
                            </PlaceButton>
                        ))}
                    </PlacesGrid>
                </PopularPlaces>

                {/* YOUR ORIGINAL ERROR MESSAGE - CLEAN STYLING */}
                {error && <ErrorMessage>{error}</ErrorMessage>}

                {/* YOUR ORIGINAL SELECTED LOCATION DISPLAY - CLEAN STYLING */}
                {selectedLocation && (
                    <>
                        <MapContainer>
                            <MapFrame
                                src={getMapUrl(selectedLocation)}
                                title={`Map of ${selectedLocation.name}`}
                                loading="lazy"
                            />
                        </MapContainer>

                        <LocationInfo>
                            <LocationTitle>
                                <FiMapPin />
                                {selectedLocation.name}
                            </LocationTitle>
                            <LocationDetails>
                                {selectedLocation.type && (
                                    <div>Type: {selectedLocation.type}</div>
                                )}
                                {selectedLocation.coords && (
                                    <div>
                                        Coordinates: {selectedLocation.coords.lat.toFixed(3)}°N, {selectedLocation.coords.lng.toFixed(3)}°E
                                    </div>
                                )}
                                {selectedLocation.country && (
                                    <div>Country: {selectedLocation.country}</div>
                                )}
                                {selectedLocation.description && (
                                    <div style={{ marginTop: '0.5rem' }}>
                                        {selectedLocation.description}
                                    </div>
                                )}
                            </LocationDetails>

                            <DirectionsButton onClick={openDirections}>
                                <FiNavigation />
                                Get Directions
                            </DirectionsButton>
                        </LocationInfo>
                    </>
                )}
            </Content>
        </Container>
    );
};

export default MapWidget;