import React from "react";

interface FileUploadProps {
  onFileSelected: (file: File | null) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelected }) => {
  const inputRef = React.useRef<HTMLInputElement | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0] ?? null;
    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  return (
    <div className="space-y-2">
      <div
        className="border-2 border-dashed border-slate-600 rounded-xl p-6 text-center cursor-pointer bg-slate-800/60 hover:bg-slate-800 transition-colors"
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <p className="font-semibold mb-1">העלה תוכנית PDF</p>
        <p className="text-sm text-slate-300">
          גרור קובץ לכאן או לחץ לבחירה
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={handleChange}
      />
    </div>
  );
};

