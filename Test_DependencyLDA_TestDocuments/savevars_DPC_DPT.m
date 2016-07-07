function savevars_DPC_DPT( savefilenm , DPC , DPT)
%  savevars_DPC_DPT( savefilenm , DPC , DPT);
if nargin==3
    save(savefilenm , 'DPC','DPT');
else
    error('Too Many Inputs?')
end

