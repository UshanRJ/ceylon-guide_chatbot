// frontend/src/components/ToolOutputs.jsx
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiMapPin, FiDollarSign, FiCloud, FiGlobe, FiExternalLink, FiChevronDown, FiChevronUp } from 'react-icons/fi';
import MapViewer from './MapViewer';
import chatService from '../services/chatService';

const ToolOutputContainer = styled.div`
  margin: 0.5rem 0;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
`;

const ToolHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  cursor: pointer;
  transition: background 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
  }
`;

const ToolTitle = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: white;
  font-weight: 600;
`;

const ToolContent = styled.div`
  padding: 1rem;
  color: white;
  display: ${props => props.isExpanded ? 'block' : 'none'};
`;

const CurrencyDisplay = styled.div`
  background: linear-gradient(135deg, #28a745, #20c997);
  border-radius: 10px;
  padding: 1.5rem;
  text-align: center;
  color: white;
`;

const CurrencyAmount = styled.div`
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
`;

const CurrencyLabel = styled.div`
  font-size: 1rem;
  opacity: 0.9;
  margin-bottom: 1rem;
`;

const ExchangeRate = styled.div`
  font-size: 0.9rem;
  opacity: 0.8;
`;

const WeatherDisplay = styled.div`
  background: linear-gradient(135deg, #007bff, #6610f2);
  border-radius: 10px;
  padding: 1.5rem;
  color: white;
`;

const WeatherMain = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const Temperature = styled.div`
  font-size: 3rem;
  font-weight: bold;
`;

const WeatherDescription = styled.div`
  font-size: 1.2rem;
  text-transform: capitalize;
`;

const WeatherDetails = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
`;

const WeatherDetail = styled.div`
  text-align: center;
  padding: 0.5rem;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
`;

const DetailLabel = styled.div`
  font-size: 0.8rem;
  opacity: 0.8;
  margin-bottom: 0.25rem;
`;

const DetailValue = styled.div`
  font-size: 1.1rem;
  font-weight: 600;
`;

const TranslationDisplay = styled.div`
  background: linear-gradient(135deg, #6f42c1, #e83e8c);
  border-radius: 10px;
  padding: 1.5rem;
  color: white;
`;

const TranslationPair = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const TranslationItem = styled.div`
  background: rgba(255, 255, 255, 0.1);
  padding: 1rem;
  border-radius: 8px;
`;

const TranslationLabel = styled.div`
  font-size: 0.9rem;
  opacity: 0.8;
  margin-bottom: 0.5rem;
`;

const TranslationText = styled.div`
  font-size: 1.2rem;
  font-weight: 600;
`;

const LocationDisplay = styled.div`
  background: linear-gradient(135deg, #fd7e14, #dc3545);
  border-radius: 10px;
  padding: 1.5rem;
  color: white;
`;

const LocationName = styled.div`
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 1rem;
`;

const LocationDetails = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
`;

const LocationDetail = styled.div`
  background: rgba(255, 255, 255, 0.1);
  padding: 0.75rem;
  border-radius: 8px;
`;

const MapContainer = styled.div`
  margin-top: 1rem;
  border-radius: 10px;
  overflow: hidden;
  height: 300px;
`;

const ExternalLink = styled.a`
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: white;
  text-decoration: none;
  background: rgba(255, 255, 255, 0.2);
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
  transition: background 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.3);
  }
`;

const ErrorDisplay = styled.div`
  background: #dc3545;
  color: white;
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
`;

const ToolOutputs = ({ tools, entities }) => {
    const [expandedTools, setExpandedTools] = useState(new Set());
    const [toolData, setToolData] = useState({});

    React.useEffect(() => {
        // Fetch tool data for each tool
        const fetchToolData = async () => {
            const data = {};

            for (const tool of tools) {
                try {
                    switch (tool) {
                        case 'currency_converter':
                            const currencyInfo = extractCurrencyInfo(entities);
                            if (currencyInfo) {
                                const result = await chatService.convertCurrency(
                                    currencyInfo.amount,
                                    currencyInfo.from_currency,
                                    currencyInfo.to_currency
                                );
                                data[tool] = result;
                            }
                            break;

                        case 'weather_checker':
                            const location = extractLocation(entities);
                            if (location) {
                                const result = await chatService.getCurrentWeather(location);
                                data[tool] = result;
                            }
                            break;

                        case 'translator':
                            const translationInfo = extractTranslationInfo(entities);
                            if (translationInfo) {
                                const result = await chatService.translateText(
                                    translationInfo.text,
                                    translationInfo.from_language,
                                    translationInfo.to_language
                                );
                                data[tool] = result;
                            }
                            break;

                        case 'maps_integration':
                            const mapLocation = extractLocation(entities);
                            if (mapLocation) {
                                const result = await chatService.getLocationInfo(mapLocation);
                                data[tool] = result;
                            }
                            break;

                        default:
                            break;
                    }
                } catch (error) {
                    console.error(`Error fetching data for ${tool}:`, error);
                    data[tool] = { success: false, error: error.message };
                }
            }

            setToolData(data);
            // Auto-expand all tools initially
            setExpandedTools(new Set(tools));
        };

        if (tools && tools.length > 0) {
            fetchToolData();
        }
    }, [tools, entities]);

    const toggleTool = (tool) => {
        const newExpanded = new Set(expandedTools);
        if (newExpanded.has(tool)) {
            newExpanded.delete(tool);
        } else {
            newExpanded.add(tool);
        }
        setExpandedTools(newExpanded);
    };

    const extractCurrencyInfo = (entities) => {
        const amounts = entities.amounts || [];
        const currencies = entities.currencies || [];

        return {
            amount: amounts.length > 0 ? parseFloat(amounts[0].text) : 1,
            from_currency: currencies.length > 0 ? currencies[0].type : 'usd',
            to_currency: currencies.length > 1 ? currencies[1].type : 'lkr'
        };
    };

    const extractLocation = (entities) => {
        const locations = entities.locations || [];
        return locations.length > 0 ? locations[0].text : 'Colombo';
    };

    const extractTranslationInfo = (entities) => {
        const languages = entities.languages || [];
        return {
            text: 'hello', // This would be extracted from the original message
            from_language: 'english',
            to_language: languages.length > 0 ? languages[0].text : 'sinhala'
        };
    };

    const getToolIcon = (tool) => {
        switch (tool) {
            case 'currency_converter':
                return <FiDollarSign />;
            case 'weather_checker':
                return <FiCloud />;
            case 'translator':
                return <FiGlobe />;
            case 'maps_integration':
                return <FiMapPin />;
            default:
                return <FiExternalLink />;
        }
    };

    const getToolTitle = (tool) => {
        switch (tool) {
            case 'currency_converter':
                return 'Currency Conversion';
            case 'weather_checker':
                return 'Weather Information';
            case 'translator':
                return 'Translation';
            case 'maps_integration':
                return 'Location & Maps';
            default:
                return tool.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    };

    const renderToolContent = (tool, data) => {
        if (!data) return <div>Loading...</div>;

        if (!data.success) {
            return <ErrorDisplay>Error: {data.error || 'Unknown error occurred'}</ErrorDisplay>;
        }

        switch (tool) {
            case 'currency_converter':
                return (
                    <CurrencyDisplay>
                        <CurrencyAmount>
                            {data.converted_amount} {data.to_currency?.toUpperCase()}
                        </CurrencyAmount>
                        <CurrencyLabel>
                            {data.amount} {data.from_currency?.toUpperCase()}
                        </CurrencyLabel>
                        <ExchangeRate>
                            Exchange Rate: 1 {data.from_currency?.toUpperCase()} = {data.exchange_rate} {data.to_currency?.toUpperCase()}
                        </ExchangeRate>
                        {data.note && (
                            <div style={{ marginTop: '1rem', fontSize: '0.9rem', opacity: 0.8 }}>
                                {data.note}
                            </div>
                        )}
                    </CurrencyDisplay>
                );

            case 'weather_checker':
                return (
                    <WeatherDisplay>
                        <WeatherMain>
                            <div>
                                <Temperature>{data.temperature}°C</Temperature>
                                <WeatherDescription>{data.description}</WeatherDescription>
                            </div>
                        </WeatherMain>
                        <WeatherDetails>
                            <WeatherDetail>
                                <DetailLabel>Humidity</DetailLabel>
                                <DetailValue>{data.humidity}%</DetailValue>
                            </WeatherDetail>
                            <WeatherDetail>
                                <DetailLabel>Wind Speed</DetailLabel>
                                <DetailValue>{data.wind_speed} km/h</DetailValue>
                            </WeatherDetail>
                            {data.feels_like && (
                                <WeatherDetail>
                                    <DetailLabel>Feels Like</DetailLabel>
                                    <DetailValue>{data.feels_like}°C</DetailValue>
                                </WeatherDetail>
                            )}
                            {data.pressure && (
                                <WeatherDetail>
                                    <DetailLabel>Pressure</DetailLabel>
                                    <DetailValue>{data.pressure} hPa</DetailValue>
                                </WeatherDetail>
                            )}
                        </WeatherDetails>
                        {data.note && (
                            <div style={{ marginTop: '1rem', fontSize: '0.9rem', opacity: 0.8 }}>
                                {data.note}
                            </div>
                        )}
                    </WeatherDisplay>
                );

            case 'translator':
                return (
                    <TranslationDisplay>
                        <TranslationPair>
                            <TranslationItem>
                                <TranslationLabel>
                                    Original ({data.from_language?.charAt(0).toUpperCase() + data.from_language?.slice(1)})
                                </TranslationLabel>
                                <TranslationText>{data.original_text}</TranslationText>
                            </TranslationItem>
                            <TranslationItem>
                                <TranslationLabel>
                                    Translation ({data.to_language?.charAt(0).toUpperCase() + data.to_language?.slice(1)})
                                </TranslationLabel>
                                <TranslationText>{data.translated_text}</TranslationText>
                            </TranslationItem>
                        </TranslationPair>
                        {data.note && (
                            <div style={{ marginTop: '1rem', fontSize: '0.9rem', opacity: 0.8 }}>
                                {data.note}
                            </div>
                        )}
                    </TranslationDisplay>
                );

            case 'maps_integration':
                return (
                    <LocationDisplay>
                        <LocationName>{data.location}</LocationName>
                        <LocationDetails>
                            {data.coordinates && (
                                <>
                                    <LocationDetail>
                                        <DetailLabel>Latitude</DetailLabel>
                                        <DetailValue>{data.coordinates.lat}</DetailValue>
                                    </LocationDetail>
                                    <LocationDetail>
                                        <DetailLabel>Longitude</DetailLabel>
                                        <DetailValue>{data.coordinates.lng}</DetailValue>
                                    </LocationDetail>
                                </>
                            )}
                            {data.country && (
                                <LocationDetail>
                                    <DetailLabel>Country</DetailLabel>
                                    <DetailValue>{data.country}</DetailValue>
                                </LocationDetail>
                            )}
                        </LocationDetails>
                        {data.coordinates && (
                            <MapContainer>
                                <MapViewer
                                    lat={data.coordinates.lat}
                                    lng={data.coordinates.lng}
                                    locationName={data.location}
                                />
                            </MapContainer>
                        )}
                        {data.map_url && (
                            <ExternalLink href={data.map_url} target="_blank" rel="noopener noreferrer">
                                <FiExternalLink />
                                View on Google Maps
                            </ExternalLink>
                        )}
                    </LocationDisplay>
                );

            default:
                return (
                    <div>
                        <pre>{JSON.stringify(data, null, 2)}</pre>
                    </div>
                );
        }
    };

    if (!tools || tools.length === 0) {
        return null;
    }

    return (
        <div>
            {tools.map((tool, index) => (
                <ToolOutputContainer key={index}>
                    <ToolHeader onClick={() => toggleTool(tool)}>
                        <ToolTitle>
                            {getToolIcon(tool)}
                            {getToolTitle(tool)}
                        </ToolTitle>
                        {expandedTools.has(tool) ? <FiChevronUp /> : <FiChevronDown />}
                    </ToolHeader>
                    <ToolContent isExpanded={expandedTools.has(tool)}>
                        {renderToolContent(tool, toolData[tool])}
                    </ToolContent>
                </ToolOutputContainer>
            ))}
        </div>
    );
};

export default ToolOutputs;