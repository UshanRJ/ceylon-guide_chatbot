// frontend/src/components/tools/CurrencyConverter.jsx - SIMPLE ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { FiRefreshCw, FiArrowRight, FiX, FiDollarSign } from 'react-icons/fi';
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
    color: #10b981;
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

// Clean converter card
const ConverterCard = styled.div`
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  animation: fadeIn 0.3s ease-out;
`;

const InputGroup = styled.div`
  margin-bottom: 1rem;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const Label = styled.label`
  display: block;
  font-size: 0.875rem;
  color: #374151;
  margin-bottom: 0.5rem;
  font-weight: 600;
`;

const AmountInput = styled.input`
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 1rem;
  transition: border-color 0.2s ease;
  
  &:focus {
    outline: none;
    border-color: #10b981;
  }
  
  &::placeholder {
    color: #94a3b8;
  }
`;

const CurrencySelect = styled.select`
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 1rem;
  background: white;
  cursor: pointer;
  transition: border-color 0.2s ease;
  
  &:focus {
    outline: none;
    border-color: #10b981;
  }
`;

const ConvertButton = styled.button`
  width: 100%;
  padding: 0.75rem 1.5rem;
  background: #10b981;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
  transition: background 0.2s ease;
  
  &:hover:not(:disabled) {
    background: #059669;
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

// Clean result display
const Result = styled.div`
  background: #ecfdf5;
  border: 2px solid #10b981;
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
  animation: fadeIn 0.3s ease-out;
`;

const ResultAmount = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: #059669;
  margin-bottom: 0.5rem;
`;

const ExchangeRate = styled.div`
  font-size: 0.875rem;
  color: #374151;
  font-weight: 500;
`;

const ResultNote = styled.div`
  margin-top: 0.75rem;
  font-size: 0.75rem;
  color: #6b7280;
  font-style: italic;
`;

// Quick amounts section
const QuickAmounts = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
  flex-wrap: wrap;
`;

const QuickAmountButton = styled.button`
  padding: 0.5rem 0.75rem;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #10b981;
    color: white;
    border-color: #10b981;
    transform: translateY(-1px);
  }
`;

// Simple swap button
const SwapButton = styled.button`
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 0.75rem 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin: 1rem auto;
  transition: all 0.2s ease;
  font-weight: 500;
  color: #374151;
  
  &:hover {
    background: #e2e8f0;
  }
  
  svg {
    transition: transform 0.2s ease;
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

// Enhanced CurrencyConverter Component with Simple Design
const CurrencyConverter = ({ onClose }) => { // NEW: Accept onClose prop
    // YOUR ORIGINAL STATE - COMPLETELY PRESERVED
    const [amount, setAmount] = useState('100');
    const [fromCurrency, setFromCurrency] = useState('USD');
    const [toCurrency, setToCurrency] = useState('LKR');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [currencies, setCurrencies] = useState([]);

    // YOUR ORIGINAL QUICK AMOUNTS - COMPLETELY PRESERVED
    const quickAmounts = [50, 100, 500, 1000];

    // YOUR ORIGINAL useEffect - COMPLETELY PRESERVED
    useEffect(() => {
        fetchSupportedCurrencies();
    }, []);

    // YOUR ORIGINAL FETCH CURRENCIES - COMPLETELY PRESERVED
    const fetchSupportedCurrencies = async () => {
        try {
            const response = await chatService.getSupportedCurrencies();
            if (response.currencies) {
                setCurrencies(response.currencies);
            }
        } catch (error) {
            // Use default currencies if API fails
            setCurrencies([
                { code: 'USD', name: 'US Dollar' },
                { code: 'EUR', name: 'Euro' },
                { code: 'GBP', name: 'British Pound' },
                { code: 'LKR', name: 'Sri Lankan Rupee' },
                { code: 'INR', name: 'Indian Rupee' },
                { code: 'AUD', name: 'Australian Dollar' },
                { code: 'CAD', name: 'Canadian Dollar' },
                { code: 'SGD', name: 'Singapore Dollar' },
                { code: 'JPY', name: 'Japanese Yen' },
                { code: 'CNY', name: 'Chinese Yuan' }
            ]);
        }
    };

    // YOUR ORIGINAL CONVERT HANDLER - COMPLETELY PRESERVED
    const handleConvert = async () => {
        if (!amount || parseFloat(amount) <= 0) return;

        setLoading(true);
        try {
            const response = await chatService.convertCurrency(
                parseFloat(amount),
                fromCurrency,
                toCurrency
            );
            if (response.success) {
                setResult(response);
            }
        } catch (error) {
            console.error('Conversion error:', error);
        } finally {
            setLoading(false);
        }
    };

    // YOUR ORIGINAL SWAP HANDLER - COMPLETELY PRESERVED
    const handleSwap = () => {
        setFromCurrency(toCurrency);
        setToCurrency(fromCurrency);
        setResult(null);
    };

    // YOUR ORIGINAL QUICK AMOUNT HANDLER - COMPLETELY PRESERVED
    const handleQuickAmount = (value) => {
        setAmount(value.toString());
    };

    return (
        <Container>
            {/* NEW: Simple Header with Close Button */}
            <Header>
                <Title>
                    <FiDollarSign />
                    Currency Converter
                </Title>
                {onClose && (
                    <CloseButton
                        onClick={onClose}
                        aria-label="Close currency converter"
                        title="Close"
                    >
                        <FiX />
                    </CloseButton>
                )}
            </Header>

            <Content>

                {result && (
                    <Result>
                        <ResultAmount>
                            {parseFloat(result.converted_amount).toLocaleString(undefined, {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2
                            })} {result.to_currency}
                        </ResultAmount>
                        <ExchangeRate>
                            1 {result.from_currency} = {parseFloat(result.exchange_rate).toLocaleString(undefined, {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 6
                        })} {result.to_currency}
                        </ExchangeRate>
                        {result.note && (
                            <ResultNote>
                                {result.note}
                            </ResultNote>
                        )}
                    </Result>
                )}
                {/* YOUR ORIGINAL CONVERTER CARD - CLEAN STYLING */}
                <ConverterCard>
                    <InputGroup>
                        <Label>Amount</Label>
                        <AmountInput
                            type="number"
                            value={amount}
                            onChange={(e) => setAmount(e.target.value)}
                            placeholder="Enter amount"
                            min="0"
                            step="0.01"
                        />
                        <QuickAmounts>
                            {quickAmounts.map((value) => (
                                <QuickAmountButton
                                    key={value}
                                    onClick={() => handleQuickAmount(value)}
                                >
                                    {value.toLocaleString()}
                                </QuickAmountButton>
                            ))}
                        </QuickAmounts>
                    </InputGroup>

                    <InputGroup>
                        <Label>From</Label>
                        <CurrencySelect
                            value={fromCurrency}
                            onChange={(e) => setFromCurrency(e.target.value)}
                        >
                            {currencies.map((currency) => (
                                <option key={currency.code} value={currency.code}>
                                    {currency.code} - {currency.name}
                                </option>
                            ))}
                        </CurrencySelect>
                    </InputGroup>

                    <SwapButton onClick={handleSwap} title="Swap currencies">
                        <FiRefreshCw />
                        Swap currencies
                    </SwapButton>

                    <InputGroup>
                        <Label>To</Label>
                        <CurrencySelect
                            value={toCurrency}
                            onChange={(e) => setToCurrency(e.target.value)}
                        >
                            {currencies.map((currency) => (
                                <option key={currency.code} value={currency.code}>
                                    {currency.code} - {currency.name}
                                </option>
                            ))}
                        </CurrencySelect>
                    </InputGroup>

                    <ConvertButton
                        onClick={handleConvert}
                        disabled={loading || !amount}
                    >
                        {loading ? <LoadingSpinner /> : <FiArrowRight />}
                        {loading ? 'Converting...' : 'Convert'}
                    </ConvertButton>
                </ConverterCard>

                {/* YOUR ORIGINAL RESULT DISPLAY - CLEAN STYLING */}
            </Content>
        </Container>
    );
};

export default CurrencyConverter;