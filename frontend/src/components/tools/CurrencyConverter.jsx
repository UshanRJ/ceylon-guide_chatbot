// frontend/src/components/tools/CurrencyConverter.jsx
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { FiRefreshCw, FiArrowRight } from 'react-icons/fi';
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

const ConverterCard = styled.div`
  background: ${props => props.theme.colors.gray[50]};
  border: 1px solid ${props => props.theme.colors.gray[200]};
  border-radius: 0.75rem;
  padding: 1.5rem;
  margin-bottom: 1rem;
`;

const InputGroup = styled.div`
  margin-bottom: 1rem;
`;

const Label = styled.label`
  display: block;
  font-size: 0.875rem;
  color: ${props => props.theme.colors.gray[700]};
  margin-bottom: 0.5rem;
  font-weight: 500;
`;

const InputRow = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const AmountInput = styled.input`
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

const CurrencySelect = styled.select`
  padding: 0.75rem 1rem;
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  font-size: 1rem;
  background: white;
  cursor: pointer;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const ConvertButton = styled.button`
  width: 100%;
  padding: 0.75rem 1.5rem;
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 0.5rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.secondary};
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const Result = styled.div`
  background: white;
  border: 2px solid ${props => props.theme.colors.primary};
  border-radius: 0.5rem;
  padding: 1.5rem;
  margin-top: 1.5rem;
  text-align: center;
`;

const ResultAmount = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: ${props => props.theme.colors.primary};
  margin-bottom: 0.5rem;
`;

const ExchangeRate = styled.div`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.gray[600]};
`;

const QuickAmounts = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
  flex-wrap: wrap;
`;

const QuickAmountButton = styled.button`
  padding: 0.25rem 0.75rem;
  background: ${props => props.theme.colors.gray[100]};
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.25rem;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const SwapButton = styled.button`
  background: ${props => props.theme.colors.gray[100]};
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  padding: 0.5rem 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 1rem auto;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.gray[200]};
  }
`;

const CurrencyConverter = () => {
    const [amount, setAmount] = useState('100');
    const [fromCurrency, setFromCurrency] = useState('USD');
    const [toCurrency, setToCurrency] = useState('LKR');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [currencies, setCurrencies] = useState([]);

    const quickAmounts = [50, 100, 500, 1000, 5000];

    useEffect(() => {
        fetchSupportedCurrencies();
    }, []);

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

    const handleSwap = () => {
        setFromCurrency(toCurrency);
        setToCurrency(fromCurrency);
        setResult(null);
    };

    const handleQuickAmount = (value) => {
        setAmount(value.toString());
    };

    return (
        <Container>
            <Title>Currency Converter</Title>

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
                                {value}
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

                <SwapButton onClick={handleSwap}>
                    <FiRefreshCw />
                    Swap Currencies
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

                <ConvertButton onClick={handleConvert} disabled={loading || !amount}>
                    <FiArrowRight />
                    {loading ? 'Converting...' : 'Convert'}
                </ConvertButton>
            </ConverterCard>

            {result && (
                <Result>
                    <ResultAmount>
                        {result.converted_amount} {result.to_currency}
                    </ResultAmount>
                    <ExchangeRate>
                        1 {result.from_currency} = {result.exchange_rate} {result.to_currency}
                    </ExchangeRate>
                    {result.note && (
                        <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#9CA3AF' }}>
                            {result.note}
                        </div>
                    )}
                </Result>
            )}
        </Container>
    );
};

export default CurrencyConverter;