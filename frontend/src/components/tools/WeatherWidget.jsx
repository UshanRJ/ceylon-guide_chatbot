// frontend/src/components/tools/WeatherWidget.jsx - SIMPLE ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiCloud, FiDroplet, FiWind, FiSearch, FiX, FiSun, FiThermometer } from 'react-icons/fi';
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
  background: #f0f9ff;
  border-bottom: 1px solid #e0f2fe;
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
    color: #0ea5e9;
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
    border-color: #0ea5e9;
  }
  
  &:disabled {
    background: #f8fafc;
    cursor: not-allowed;
  }
`;

const SearchButton = styled.button`
  padding: 0.75rem 1rem;
  background: #0ea5e9;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  transition: background 0.2s ease;
  
  &:hover:not(:disabled) {
    background: #0284c7;
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

// Clean weather card
const WeatherCard = styled.div`
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border: 1px solid #bae6fd;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  animation: fadeIn 0.3s ease-out;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
`;

const WeatherHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const CityName = styled.h3`
  font-size: 1.25rem;
  color: #1e40af;
  margin: 0;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  svg {
    color: #f59e0b;
  }
`;

const Temperature = styled.div`
  font-size: 2.5rem;
  font-weight: 700;
  color: #0369a1;
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const WeatherDescription = styled.p`
  color: #1e40af;
  text-transform: capitalize;
  margin: 0 0 1.5rem 0;
  font-size: 1.1rem;
  font-weight: 500;
  text-align: center;
  padding: 0.5rem;
  background: rgba(255, 255, 255, 0.5);
  border-radius: 8px;
`;

const WeatherDetails = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
`;

const DetailItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: #1e40af;
  background: rgba(255, 255, 255, 0.6);
  padding: 0.75rem;
  border-radius: 8px;
  font-weight: 500;
  
  svg {
    color: #0ea5e9;
    flex-shrink: 0;
  }
`;

// Simple error message
const ErrorMessage = styled.div`
  background: #fef2f2;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  border: 1px solid #fecaca;
  font-weight: 500;
  animation: fadeIn 0.3s ease-out;
`;

// Simple loading message
const Loading = styled.div`
  text-align: center;
  color: #0ea5e9;
  padding: 2rem;
  font-size: 1.1rem;
  font-weight: 500;
  animation: fadeIn 0.3s ease-out;
`;

// Preset cities section
const PresetCities = styled.div`
  margin-bottom: 1.5rem;
`;

const SectionLabel = styled.div`
  font-size: 0.875rem;
  color: #64748b;
  margin-bottom: 0.75rem;
  font-weight: 600;
`;

const PresetButton = styled.button`
  padding: 0.5rem 1rem;
  margin: 0.25rem;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  color: #374151;
  
  &:hover {
    background: #0ea5e9;
    color: white;
    border-color: #0ea5e9;
    transform: translateY(-1px);
  }
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

// Enhanced WeatherWidget Component with Simple Design
const WeatherWidget = ({ onClose }) => { // NEW: Accept onClose prop
    // YOUR ORIGINAL STATE - COMPLETELY PRESERVED
    const [city, setCity] = useState('');
    const [weatherData, setWeatherData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // YOUR ORIGINAL POPULAR CITIES - COMPLETELY PRESERVED
    const popularCities = ['Colombo', 'Kandy', 'Galle', 'Negombo', 'Ella', 'Sigiriya'];

    // YOUR ORIGINAL FETCH WEATHER - COMPLETELY PRESERVED
    const fetchWeather = async (cityName) => {
        setLoading(true);
        setError(null);

        try {
            const response = await chatService.getCurrentWeather(cityName);
            if (response.success) {
                setWeatherData(response);
            } else {
                setError(response.error || 'Failed to fetch weather data');
            }
        } catch (err) {
            setError('Failed to connect to weather service');
        } finally {
            setLoading(false);
        }
    };

    // YOUR ORIGINAL SUBMIT HANDLER - COMPLETELY PRESERVED
    const handleSubmit = (e) => {
        e.preventDefault();
        if (city.trim()) {
            fetchWeather(city);
        }
    };

    // YOUR ORIGINAL PRESET CLICK HANDLER - COMPLETELY PRESERVED
    const handlePresetClick = (cityName) => {
        setCity(cityName);
        fetchWeather(cityName);
    };

    return (
        <Container>
            {/* NEW: Simple Header with Close Button */}
            <Header>
                <Title>
                    <FiSun />
                    Weather Information
                </Title>
                {onClose && (
                    <CloseButton
                        onClick={onClose}
                        aria-label="Close weather widget"
                        title="Close"
                    >
                        <FiX />
                    </CloseButton>
                )}
            </Header>

            <Content>
                {/* YOUR ORIGINAL SEARCH FORM - CLEAN STYLING */}
                <SearchForm onSubmit={handleSubmit}>
                    <SearchInput
                        type="text"
                        value={city}
                        onChange={(e) => setCity(e.target.value)}
                        placeholder="Enter city name..."
                        disabled={loading}
                    />
                    <SearchButton
                        type="submit"
                        disabled={loading || !city.trim()}
                    >
                        {loading ? <LoadingSpinner /> : <FiSearch />}
                        Search
                    </SearchButton>
                </SearchForm>

                {/* YOUR ORIGINAL PRESET CITIES - CLEAN STYLING */}
                <PresetCities>
                    <SectionLabel>
                        Popular cities in Sri Lanka:
                    </SectionLabel>
                    {popularCities.map((cityName) => (
                        <PresetButton
                            key={cityName}
                            onClick={() => handlePresetClick(cityName)}
                        >
                            {cityName}
                        </PresetButton>
                    ))}
                </PresetCities>

                {/* YOUR ORIGINAL LOADING STATE - CLEAN STYLING */}
                {loading && <Loading>Loading weather data...</Loading>}

                {/* YOUR ORIGINAL ERROR MESSAGE - CLEAN STYLING */}
                {error && <ErrorMessage>{error}</ErrorMessage>}

                {/* YOUR ORIGINAL WEATHER DISPLAY - ENHANCED STYLING */}
                {weatherData && !loading && (
                    <WeatherCard>
                        <WeatherHeader>
                            <CityName>
                                <FiSun />
                                {weatherData.location}
                            </CityName>
                            <Temperature>
                                <FiThermometer />
                                {weatherData.temperature}°C
                            </Temperature>
                        </WeatherHeader>

                        <WeatherDescription>
                            {weatherData.description}
                        </WeatherDescription>

                        <WeatherDetails>
                            <DetailItem>
                                <FiThermometer />
                                <span>Feels like: {weatherData.feels_like}°C</span>
                            </DetailItem>
                            <DetailItem>
                                <FiDroplet />
                                <span>Humidity: {weatherData.humidity}%</span>
                            </DetailItem>
                            <DetailItem>
                                <FiWind />
                                <span>Wind: {weatherData.wind_speed} km/h</span>
                            </DetailItem>
                            <DetailItem>
                                <FiCloud />
                                <span>Pressure: {weatherData.pressure} hPa</span>
                            </DetailItem>
                        </WeatherDetails>
                    </WeatherCard>
                )}
            </Content>
        </Container>
    );
};

export default WeatherWidget;