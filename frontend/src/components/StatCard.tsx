import React from 'react';

type AccentColor = 'orange' | 'green' | 'blue' | 'amber' | 'red' | 'navy';

interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  accent?: AccentColor;
  icon?: React.ReactNode;
}

const accentMap: Record<AccentColor, string> = {
  orange: 'var(--orange)',
  green:  'var(--green)',
  blue:   '#1d4ed8',
  amber:  'var(--amber)',
  red:    'var(--red)',
  navy:   'var(--navy)',
};

const StatCard: React.FC<StatCardProps> = ({ label, value, sub, accent = 'navy', icon }) => (
  <div style={{
    background: '#fff',
    borderRadius: 'var(--r)',
    boxShadow: 'var(--sh1)',
    borderTop: `3px solid ${accentMap[accent]}`,
    padding: '14px 18px',
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  }}>
    {icon && <div style={{ marginBottom: 4, color: accentMap[accent] }}>{icon}</div>}
    <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--s900)', letterSpacing: '-1px', lineHeight: 1 }}>
      {value}
    </div>
    <div style={{ fontSize: '.82rem', color: 'var(--s500)', fontWeight: 500 }}>{label}</div>
    {sub && <div style={{ fontSize: '.75rem', color: 'var(--s400)' }}>{sub}</div>}
  </div>
);

export default StatCard;
