import React from 'react';

interface ProgressBarProps {
  percent: number;
  height?: number;
  showLabel?: boolean;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ percent, height = 10, showLabel = false }) => {
  const clamped = Math.min(100, Math.max(0, percent));
  const color =
    clamped >= 60 ? 'var(--green)' :
    clamped >= 30 ? 'var(--amber)' :
    'var(--red)';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        flex: 1,
        height,
        background: 'var(--s200)',
        borderRadius: 99,
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${clamped}%`,
          height: '100%',
          background: color,
          borderRadius: 99,
          transition: 'width .3s ease',
        }} />
      </div>
      {showLabel && (
        <span style={{ fontSize: '.78rem', fontWeight: 600, color: 'var(--s500)', minWidth: 32, textAlign: 'left' }}>
          {clamped}%
        </span>
      )}
    </div>
  );
};

export default ProgressBar;
