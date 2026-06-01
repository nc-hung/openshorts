import React, { useState, useEffect } from 'react';
import { Youtube, Upload, FileVideo, X, ChevronDown, ChevronUp } from 'lucide-react';
import { getApiUrl } from '../config';

const LANGUAGES = [
  { value: 'vi', label: 'Tiếng Việt' },
  { value: 'en', label: 'English' },
  { value: 'auto', label: 'Auto-detect' },
];

const ASR_PROVIDERS = [
  { value: 'auto', label: 'Local + fallback' },
  { value: 'local', label: 'Local only' },
  { value: 'google', label: 'Google Cloud STT' },
];

export default function MediaInput({ onProcess, isProcessing }) {
  const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
  const [mode, setMode] = useState('url');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [language, setLanguage] = useState('vi');
  const [asrProvider, setAsrProvider] = useState('auto');
  const [asrHotwords, setAsrHotwords] = useState('');

  useEffect(() => {
    fetch(getApiUrl('/api/config'))
      .then((r) => r.ok ? r.json() : null)
      .then((cfg) => {
        if (cfg && cfg.youtubeUrlEnabled === false) {
          setYoutubeUrlEnabled(false);
          setMode('file');
        }
      })
      .catch(() => {});
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!acknowledged) return;
    const payload = {
      type: mode,
      payload: mode === 'url' ? url : file,
      acknowledged: true,
      language: language, // "auto" → service passes None to whisper for auto-detect
      asrProvider: asrProvider === 'auto' ? undefined : asrProvider,
      asrHotwords: asrHotwords.trim() || undefined,
    };
    if (mode === 'url' && url) {
      onProcess(payload);
    } else if (mode === 'file' && file) {
      onProcess(payload);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setMode('file');
    }
  };

  return (
    <div className="bg-surface border border-white/5 rounded-2xl p-6 animate-[fadeIn_0.6s_ease-out]">
      <div className="flex gap-4 mb-6 border-b border-white/5 pb-4">
        {youtubeUrlEnabled && (
          <button
            onClick={() => setMode('url')}
            className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'url'
              ? 'text-primary border-b-2 border-primary -mb-[17px]'
              : 'text-zinc-400 hover:text-white'
              }`}
          >
            <Youtube size={18} />
            YouTube URL
          </button>
        )}
        <button
          onClick={() => setMode('file')}
          className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'file'
            ? 'text-primary border-b-2 border-primary -mb-[17px]'
            : 'text-zinc-400 hover:text-white'
            }`}
        >
          <Upload size={18} />
          Upload File
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        {mode === 'url' ? (
          <div className="space-y-4">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="input-field"
              required
            />
          </div>
        ) : (
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-zinc-700 hover:border-zinc-500 bg-white/5'
              }`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="flex items-center justify-center gap-3 text-white">
                <FileVideo className="text-primary" />
                <span className="font-medium">{file.name}</span>
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="p-1 hover:bg-white/10 rounded-full"
                >
                  <X size={16} />
                </button>
              </div>
            ) : (
              <label className="cursor-pointer block">
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  className="hidden"
                />
                <Upload className="mx-auto mb-3 text-zinc-500" size={24} />
                <p className="text-zinc-400">Click to upload or drag and drop</p>
                <p className="text-xs text-zinc-600 mt-1">MP4, MOV up to 500MB</p>
              </label>
            )}
          </div>
        )}

        {/* Advanced Settings */}
        <div className="mt-4">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            Advanced settings
          </button>

          {showAdvanced && (
            <div className="mt-3 p-4 bg-white/[0.02] border border-white/5 rounded-xl space-y-4">
              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">Language</label>
                <div className="flex gap-2">
                  {LANGUAGES.map((l) => (
                    <button
                      key={l.value}
                      type="button"
                      onClick={() => setLanguage(l.value)}
                      className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                        language === l.value
                          ? 'border-primary text-primary bg-primary/10'
                          : 'border-white/10 text-zinc-400 hover:border-zinc-500'
                      }`}
                    >
                      {l.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">ASR Provider</label>
                <div className="flex gap-2">
                  {ASR_PROVIDERS.map((p) => (
                    <button
                      key={p.value}
                      type="button"
                      onClick={() => setAsrProvider(p.value)}
                      className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                        asrProvider === p.value
                          ? 'border-primary text-primary bg-primary/10'
                          : 'border-white/10 text-zinc-400 hover:border-zinc-500'
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">
                  Hotwords <span className="text-zinc-600">(comma-separated, optional)</span>
                </label>
                <input
                  type="text"
                  value={asrHotwords}
                  onChange={(e) => setAsrHotwords(e.target.value)}
                  placeholder="e.g. OpenAI, ChatGPT, API"
                  className="input-field text-sm"
                />
              </div>
            </div>
          )}
        </div>

        <label className="flex items-start gap-2 mt-5 text-xs text-zinc-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(e) => setAcknowledged(e.target.checked)}
            className="mt-0.5 accent-primary cursor-pointer"
          />
          <span>
            I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/#legal" target="_blank" rel="noopener noreferrer" className="text-primary underline" onClick={(e) => e.stopPropagation()}>Terms & Privacy</a>.
          </span>
        </label>

        <button
          type="submit"
          disabled={isProcessing || !acknowledged || (mode === 'url' && !url) || (mode === 'file' && !file)}
          className="w-full btn-primary mt-4 flex items-center justify-center gap-2"
        >
          {isProcessing ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Processing Video...
            </>
          ) : (
            <>
              Generate Clips
            </>
          )}
        </button>
      </form>
    </div>
  );
}
