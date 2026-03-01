import React from 'react';

type BadgeVariant = 'green' | 'amber' | 'red' | 'blue' | 'navy' | 'gray';

interface StatusBadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
}

const variantStyles: Record<BadgeVariant, { bg: string; color: string; dot: string }> = {
  green: { bg: 'var(--green-light, #dcfce7)', color: 'var(--green)',  dot: 'var(--green)' },
  amber: { bg: 'var(--amber-light, #fef3c7)', color: 'var(--amber)',  dot: 'var(--amber)' },
  red:   { bg: 'var(--red-light,   #fee2e2)', color: 'var(--red)',    dot: 'var(--red)'   },
  blue:  { bg: '#dbeafe',                     color: '#1d4ed8',        dot: '#1d4ed8'      },
  navy:  { bg: '#e0eaf5',                     color: 'var(--navy)',    dot: 'var(--navy)'  },
  gray:  { bg: 'var(--s100)',                  color: 'var(--s500)',    dot: 'var(--s400)'  },
};

const StatusBadge: React.FC<StatusBadgeProps> = ({ variant = 'gray', children }) => {
  const s = variantStyles[variant];
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      padding: '3px 10px',
      borderRadius: 20,
      fontSize: '.78rem',
      fontWeight: 600,
      background: s.bg,
      color: s.color,
      whiteSpace: 'nowrap',
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: s.dot, flexShrink: 0 }} />
      {children}
    </span>
  );
};

export default StatusBadge;
