function savevars_Z_ZZ_FinalStates( savefilenm , Z , ZZ)
%  savevars_Final_Z_States( savefilenm , Z , ZZ)
if nargin==3
    save(savefilenm , 'Z','ZZ');
else
    error('Too Many or Too Few Inputs')
end

