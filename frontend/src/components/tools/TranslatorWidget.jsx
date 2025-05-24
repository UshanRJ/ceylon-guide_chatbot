// frontend/src/components/tools/TranslatorWidget.jsx - SIMPLE ENHANCED UI/UX WITH PRESERVED FUNCTIONALITY
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiGlobe, FiRefreshCw, FiCopy, FiCheck, FiX, FiMessageSquare } from 'react-icons/fi';
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
  background: #fef3f2;
  border-bottom: 1px solid #fecaca;
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
    color: #dc2626;
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

// Clean translator card
const TranslatorCard = styled.div`
  background: #fef9f5;
  border: 1px solid #fed7aa;
  border-radius: 8px;
  padding: 1.5rem;
  animation: fadeIn 0.3s ease-out;
`;

// Language selector section
const LanguageSelector = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const LanguageSelect = styled.select`
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 1rem;
  background: white;
  cursor: pointer;
  transition: border-color 0.2s ease;
  
  &:focus {
    outline: none;
    border-color: #dc2626;
  }
`;

const SwapButton = styled.button`
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 0.75rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: all 0.2s ease;
  color: #374151;
  
  &:hover {
    background: #e2e8f0;
    transform: rotate(180deg);
  }
  
  svg {
    transition: transform 0.2s ease;
  }
`;

const TextArea = styled.textarea`
  width: 100%;
  padding: 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 1rem;
  resize: vertical;
  min-height: 120px;
  transition: border-color 0.2s ease;
  font-family: inherit;
  
  &::placeholder {
    color: #94a3b8;
  }
  
  &:focus {
    outline: none;
    border-color: #dc2626;
  }
`;

const TranslateButton = styled.button`
  width: 100%;
  padding: 0.75rem 1.5rem;
  background: #dc2626;
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
  margin: 1rem 0;
  transition: background 0.2s ease;
  
  &:hover:not(:disabled) {
    background: #b91c1c;
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

// Clean result display
const ResultCard = styled.div`
  background: #f0fdf4;
  border: 2px solid #22c55e;
  border-radius: 8px;
  padding: 1rem;
  margin-top: 1rem;
  position: relative;
  animation: fadeIn 0.3s ease-out;
`;

const ResultText = styled.div`
  font-size: 1.125rem;
  color: #1f2937;
  line-height: 1.6;
  padding-right: 3rem;
  font-weight: 500;
`;

const CopyButton = styled.button`
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: ${props => props.$copied ? '#22c55e' : '#f1f5f9'};
  color: ${props => props.$copied ? 'white' : '#374151'};
  border: 1px solid ${props => props.$copied ? '#22c55e' : '#cbd5e1'};
  border-radius: 6px;
  padding: 0.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.$copied ? '#16a34a' : '#e2e8f0'};
    transform: scale(1.05);
  }
`;

// Common phrases section
const CommonPhrases = styled.div`
  margin-top: 2rem;
`;

const PhrasesTitle = styled.h3`
  font-size: 1.125rem;
  color: #1f2937;
  margin-bottom: 1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  svg {
    color: #dc2626;
  }
`;

const PhrasesList = styled.div`
  display: grid;
  gap: 0.75rem;
`;

const PhraseItem = styled.button`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 0.875rem;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #fef9f5;
    border-color: #dc2626;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1);
  }
  
  .english {
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.25rem;
  }
  
  .translation {
    font-size: 0.875rem;
    color: #6b7280;
    font-weight: 500;
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

// Enhanced TranslatorWidget Component with Simple Design
const TranslatorWidget = ({ onClose }) => { // NEW: Accept onClose prop
    // YOUR ORIGINAL STATE - COMPLETELY PRESERVED
    const [sourceText, setSourceText] = useState('');
    const [translatedText, setTranslatedText] = useState('');
    const [sourceLang, setSourceLang] = useState('english');
    const [targetLang, setTargetLang] = useState('sinhala');
    const [loading, setLoading] = useState(false);
    const [copied, setCopied] = useState(false);

    // YOUR ORIGINAL LANGUAGES - COMPLETELY PRESERVED
    const languages = [
        { code: 'english', name: 'English' },
        { code: 'sinhala', name: 'සිංහල (Sinhala)' },
        { code: 'tamil', name: 'தமிழ் (Tamil)' }
    ];

    // YOUR ORIGINAL COMMON PHRASES - COMPLETELY PRESERVED
    const commonPhrases = [
        { en: 'Hello', si: 'ආයුබෝවන් (Ayubowan)', ta: 'வணக்கம் (Vanakkam)' },
        { en: 'Thank you', si: 'ස්තූතියි (Sthuthiyi)', ta: 'நன்றி (Nandri)' },
        { en: 'How are you?', si: 'කොහොමද? (Kohomada?)', ta: 'எப்படி இருக்கிறீர்கள்? (Eppadi irukkirirgal?)' },
        { en: 'Where is...?', si: 'කොහෙද...? (Koheda...?)', ta: 'எங்கே...? (Enge...?)' },
        { en: 'How much?', si: 'කීයද? (Kiyada?)', ta: 'எவ்வளவு? (Evvalavu?)' }
    ];

    // YOUR ORIGINAL TRANSLATE HANDLER - COMPLETELY PRESERVED
    const handleTranslate = async () => {
        if (!sourceText.trim()) return;

        setLoading(true);
        try {
            const response = await chatService.translateText(
                sourceText,
                sourceLang,
                targetLang
            );
            if (response.success) {
                setTranslatedText(response.translated_text);
            }
        } catch (error) {
            console.error('Translation error:', error);
        } finally {
            setLoading(false);
        }
    };

    // YOUR ORIGINAL SWAP HANDLER - COMPLETELY PRESERVED
    const handleSwap = () => {
        setSourceLang(targetLang);
        setTargetLang(sourceLang);
        setSourceText(translatedText);
        setTranslatedText(sourceText);
    };

    // YOUR ORIGINAL COPY HANDLER - COMPLETELY PRESERVED
    const handleCopy = () => {
        navigator.clipboard.writeText(translatedText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    // YOUR ORIGINAL PHRASE CLICK HANDLER - COMPLETELY PRESERVED
    const handlePhraseClick = (phrase) => {
        setSourceText(phrase.en);
        setSourceLang('english');

        if (targetLang === 'sinhala') {
            setTranslatedText(phrase.si);
        } else if (targetLang === 'tamil') {
            setTranslatedText(phrase.ta);
        }
    };

    return (
        <Container>
            {/* NEW: Simple Header with Close Button */}
            <Header>
                <Title>
                    <FiGlobe />
                    Translator
                </Title>
                {onClose && (
                    <CloseButton
                        onClick={onClose}
                        aria-label="Close translator widget"
                        title="Close"
                    >
                        <FiX />
                    </CloseButton>
                )}
            </Header>

            <Content>
                {/* YOUR ORIGINAL TRANSLATOR CARD - CLEAN STYLING */}
                <TranslatorCard>
                    <LanguageSelector>
                        <LanguageSelect
                            value={sourceLang}
                            onChange={(e) => setSourceLang(e.target.value)}
                        >
                            {languages.map((lang) => (
                                <option key={lang.code} value={lang.code}>
                                    {lang.name}
                                </option>
                            ))}
                        </LanguageSelect>

                        <SwapButton onClick={handleSwap} title="Swap languages">
                            <FiRefreshCw />
                        </SwapButton>

                        <LanguageSelect
                            value={targetLang}
                            onChange={(e) => setTargetLang(e.target.value)}
                        >
                            {languages.map((lang) => (
                                <option key={lang.code} value={lang.code}>
                                    {lang.name}
                                </option>
                            ))}
                        </LanguageSelect>
                    </LanguageSelector>

                    <TextArea
                        value={sourceText}
                        onChange={(e) => setSourceText(e.target.value)}
                        placeholder="Enter text to translate..."
                    />

                    <TranslateButton
                        onClick={handleTranslate}
                        disabled={loading || !sourceText.trim()}
                    >
                        {loading ? <LoadingSpinner /> : <FiGlobe />}
                        {loading ? 'Translating...' : 'Translate'}
                    </TranslateButton>

                    {/* YOUR ORIGINAL RESULT DISPLAY - CLEAN STYLING */}
                    {translatedText && (
                        <ResultCard>
                            <ResultText>{translatedText}</ResultText>
                            <CopyButton
                                onClick={handleCopy}
                                $copied={copied}
                                title={copied ? 'Copied!' : 'Copy translation'}
                            >
                                {copied ? <FiCheck /> : <FiCopy />}
                            </CopyButton>
                        </ResultCard>
                    )}
                </TranslatorCard>

                {/* YOUR ORIGINAL COMMON PHRASES - CLEAN STYLING */}
                <CommonPhrases>
                    <PhrasesTitle>
                        <FiMessageSquare />
                        Common Phrases
                    </PhrasesTitle>
                    <PhrasesList>
                        {commonPhrases.map((phrase, index) => (
                            <PhraseItem
                                key={index}
                                onClick={() => handlePhraseClick(phrase)}
                            >
                                <div className="english">{phrase.en}</div>
                                <div className="translation">
                                    {targetLang === 'sinhala' ? phrase.si : phrase.ta}
                                </div>
                            </PhraseItem>
                        ))}
                    </PhrasesList>
                </CommonPhrases>
            </Content>
        </Container>
    );
};

export default TranslatorWidget;