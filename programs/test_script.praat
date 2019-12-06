clearinfo

form Test command line calls
    sentence directory


wd$ = homeDirectory$ + "/Desktop/neutrality/masc_redos/"

inDir$ = wd$
inDirWavs$ = inDir$ + "*/*/*/*.wav"
appendInfoLine: inDirWavs$

wavList = Create Strings as file list: "wavList", inDirWavs$

selectObject: wavList

numFiles = Get number of strings
for fileNum from 1 to numFiles
	fileName$ = Get string: fileNum
	appendInfoLine: fileName$
endfor