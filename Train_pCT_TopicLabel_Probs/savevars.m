function savevars( savefilenm , DP , WP , Z)
%  savevars( savefilenm , DP , WP)
if nargin <= 2
    error('too few inputs')
    
elseif nargin==3
    save(savefilenm , 'DP','WP');
    
elseif nargin==4
    save(savefilenm , 'DP','WP','Z');
    
else
    error('Too Many Inputs?')
end

