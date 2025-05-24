// frontend/src/components/tools/MapWidget.jsx
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiMapPin, FiSearch, FiNavigation } from 'react-icons/fi';
import chatService from '../../services/chatService';

const Container = styled.div`
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  color: ${props => props.theme.colors.gray[900]};
  margin-bottom: 1.5rem;
`;

const SearchForm = styled.form`
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
`;

const SearchInput = styled.input`
  flex: 1;
  padding: 0.75rem 1rem;
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  font-size: 1rem;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const SearchButton = styled.button`
  padding: 0.75rem 1rem;
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    background: ${props => props.theme.colors.secondary};
  }
`;

const PopularPlaces = styled.div`
  margin-bottom: 1.5rem;
`;

const PlacesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin-top: 0.5rem;
`;

const PlaceButton = styled.button`
  padding: 0.75rem;
  background: ${props => props.theme.colors.gray[100]};
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  
  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
    border-color: ${props => props.theme.colors.primary};
  }
  
  .name {
    font-weight: 500;
    font-size: 0.875rem;
  }
  
  .type {
    font-size: 0.75rem;
    opacity: 0.8;
  }
`;

const MapContainer = styled.div`
  width: 100%;
  height: 400px;
  border-radius: 0.75rem;
  overflow: hidden;
  border: 1px solid ${props => props.theme.colors.gray[200]};
  margin-bottom: 1rem;
  background: ${props => props.theme.colors.gray[100]};
  position: relative;
`;

const MapFrame = styled.iframe`
  width: 100%;
  height: 100%;
  border: none;
`;

const LocationInfo = styled.div`
  background: white;
  border: 1px solid ${props => props.theme.colors.gray[200]};
  border-radius: 0.5rem;
  padding: 1rem;
  margin-top: 1rem;
`;

const LocationTitle = styled.h3`
  font-size: 1.125rem;
  color: ${props => props.theme.colors.gray[900]};
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const LocationDetails = styled.div`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.gray[600]};
  line-height: 1.6;
`;

const DirectionsButton = styled.button`
  width: 100%;
  padding: 0.75rem;
  background: ${props => props.theme.colors.success};
  color: white;
  border: none;
  border-radius: 0.5rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
  
  &:hover {
    background: ${props => props.theme.colors.success}DD;
  }
`;

const ErrorMessage = styled.div`
  background: ${props => props.theme.colors.danger}10;
  color: ${props => props.theme.colors.danger};
  padding: 1rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
`;

const MapWidget = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedLocation, setSelectedLocation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const popularPlaces = [
        { name: 'Sigiriya', type: 'Ancient Fortress', coords: { lat: 7.957, lng: 80.760 } },
        { name: 'Temple of Tooth', type: 'Sacred Temple', coords: { lat: 7.294, lng: 80.641 } },
        { name: 'Galle Fort', type: 'Historic Fort', coords: { lat: 6.032, lng: 80.215 } },
        { name: "Adam's Peak", type: 'Sacred Mountain', coords: { lat: 6.809, lng: 80.499 } },
        { name: 'Ella Rock', type: 'Scenic Viewpoint', coords: { lat: 6.867, lng: 81.053 } },
        { name: 'Mirissa Beach', type: 'Beach', coords: { lat: 5.945, lng: 80.459 } }
    ];

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

    const handlePlaceClick = (place) => {
        setSelectedLocation(place);
        setSearchQuery(place.name);
    };

    const getMapUrl = (location) => {
        if (!location || !location.coords) return '';

        const { lat, lng } = location.coords;
        // Using OpenStreetMap as it doesn't require API key
        return `https://www.openstreetmap.org/export/embed.html?bbox=${lng-0.01},${lat-0.01},${lng+0.01},${lat+0.01}&layer=mapnik&marker=${lat},${lng}`;
    };

    const openDirections = () => {
        if (selectedLocation && selectedLocation.coords) {
            const { lat, lng } = selectedLocation.coords;
            window.open(`https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`, '_blank');
        }
    };

    return (
        <Container>
            <Title>Maps & Locations</Title>

            <SearchForm onSubmit={handleSearch}>
                <SearchInput
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search for a location..."
                    disabled={loading}
                />
                <SearchButton type="submit" disabled={loading || !searchQuery.trim()}>
                    <FiSearch />
                    Search
                </SearchButton>
            </SearchForm>

            <PopularPlaces>
                <div style={{ fontSize: '0.875rem', color: '#6B7280', marginBottom: '0.5rem' }}>
                    Popular destinations:
                </div>
                <PlacesGrid>
                    {popularPlaces.map((place, index) => (
                        <PlaceButton key={index} onClick={() => handlePlaceClick(place)}>
                            <div className="name">{place.name}</div>
                            <div className="type">{place.type}</div>
                        </PlaceButton>
                    ))}
                </PlacesGrid>
            </PopularPlaces>

            {error && <ErrorMessage>{error}</ErrorMessage>}

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
        </Container>
    );
};

export default MapWidget;