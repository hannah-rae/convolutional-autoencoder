function danika_mcam_remove_duplicate_files,file_list
	prefixes=!NULL
	foreach file, file_list do begin
		info=mmm_io_product_name(file)
		prefixes=[prefixes,strmid(info.file_name,0,info.version_position)]
	endforeach
	if n_elements(file_list) gt 0 then return,file_list[uniq(prefixes,sort(file_list))] else return,file_list
end

function danika_mcam_prefer_lossless_files,file_list
	files_out=!NULL
	fullframes=!NULL
	prefixes=!NULL
	foreach file, file_list do begin
		info=mmm_io_product_name(file)
		if info.thumbnail then files_out=[files_out,file] else begin
			prefixes = [prefixes,info.sol+info.instrument+string(info.sequence_num,format='(I6.6)')+info.command_num+info.type]
			fullframes=[fullframes,file]
		endelse
	endforeach
	for i=0,n_elements(prefixes)-1 do begin
		if strmid(prefixes[i],0,/reverse_offset) eq 'C' then files_out=[files_out,fullframes[i]] else begin
			strlength=strlen(prefixes[i])
			if (where(strmatch(prefixes,strmid(prefixes[i],0,strlength-1)+'C')))[0] ne -1 then continue else files_out=[files_out,fullframes[i]]
		endelse
	endfor
	return,files_out[sort(files_out)]
end

;Takes an array of files, returns list where each element is an array of files with a different pointing than the previous.
function danika_mcam_resolve_multiple_pointings,file_list
	previous_instrument_azimuth=!NULL
	previous_instrument_elevation=!NULL
	fileset=list()
	set=[]
	if n_elements(file_list) eq 0 then return,fileset
	filelist=file_list[sort(file_list)]
	foreach file,filelist do begin
		if  mer_readpds(file,kv=kv,/no_image) then $
			message,'Could not read PDS label for: '+file+'!'
		if not mer_pds_lbl_kv_get_value(kv,'INSTRUMENT_AZIMUTH',$
			rover_instrument_azimuth,'DOUBLE', $
			in_block='ROVER_DERIVED_GEOMETRY_PARMS', bl_type='GROUP') then $
			message,'Could not find ROVER_DERIVED_GEOMETRY_PARMS INSTRUMENT_AZIMUTH for: '+file
		if not mer_pds_lbl_kv_get_value(kv,'INSTRUMENT_ELEVATION', $
			rover_instrument_elevation,'DOUBLE', $
			in_block='ROVER_DERIVED_GEOMETRY_PARMS', bl_type='GROUP') then $
			message,'Could not find ROVER_DERIVED_GEOMETRY_PARMS INSTRUMENT_ELEVATION for: '+file
		rover_instrument_azimuth=long(rover_instrument_azimuth*1000) ;convert to long for comparison
		rover_instrument_elevation=long(rover_instrument_elevation*1000)
		if (rover_instrument_azimuth ne previous_instrument_azimuth) || $
				(rover_instrument_elevation ne previous_instrument_elevation) then begin
			fileset.add,set
			set=[file]
			previous_instrument_azimuth=rover_instrument_azimuth
			previous_instrument_elevation=rover_instrument_elevation
		endif else set=[set,file]
	endforeach
	if strlen(set[0]) gt 0 then fileset.add,set
	if n_elements(fileset) gt 1 then fileset=fileset[1:-1] ;get rid of empty first element
	return,fileset
end

function danika_mcam_resolve_multiple_filters,file_list
	if typename(file_list) eq 'LIST' then filelist=file_list else filelist=list(file_list)
	output=list()
	for i=0, n_elements(filelist)-1 do begin
		filters=[]
		fileset=[]
		newset=[]
		foreach file,filelist[i] do begin
			if  mer_readpds(file,kv=kv,/no_image) then $
				message,'Could not read PDS label for: '+file+'!'
			if not mer_pds_lbl_kv_get_value(kv,'FILTER_NAME',$
				filter,'STRING',in_block='OBSERVATION_REQUEST_PARMS', bl_type='GROUP') then $
				message,'Could not find FILTER_NAME for: '+file
			if where(filters eq filter) eq -1 then begin 
				fileset=[fileset,file]
				filters=[filters,filter]
			endif else newset=[newset,file]
		endforeach
		output.add,fileset
		if n_elements(newset) ne 0 then output = output + danika_mcam_resolve_multiple_filters(newset)
	endfor
	return,output
end

function hanika_mcam_get_filelist,input_sol,type,sequence=sequence,thumbnails=thumbnails,both=both,ignore_focus_thumbnails=ignore_focus_thumbnails,remove_duplicates=remove_duplicates,resolve_multiple_pointings=resolve_multiple_pointings,resolve_multiple_filters=resolve_multiple_filters,prefer_lossless=prefer_lossless,ignore_old_filenames=ignore_old_filenames

	current_software_version = 'V2'
	focus_thumbnail_max_size = 32

	sol=string(input_sol,format='(I5.5)')
	
	; Be forgiving on input for sequence
	STRING = 7
	if keyword_set(sequence) && (size(sequence,/type) eq STRING) then begin
		if strmid(sequence,0,4) eq 'mcam' then sequence=strmid(sequence,4)
	endif

	case strupcase(type) of 
		'RAD': directory_list=file_search('/ods/surface/sol/'+sol+'/soas/rdr/mcam/rad/*IMG*')
		'IOF': directory_list=file_search('/ods/surface/sol/'+sol+'/soas/rdr/mcam/rad/iof/*IMG*')
		'EDR': directory_list=file_search('/ods/surface/sol/'+sol+'/soas/edr/mcam/*IMG*')
		else: message,'Keyword TYPE must be one of EDR, RAD, or IOF!'
	endcase
	
	if keyword_set(both) then thumbnails=1

	files_out=[]
	foreach file,directory_list do begin
		info=mmm_io_product_name(file)
		if keyword_set(ignore_old_filenames) && info.name_version ne 'v2' then continue
		if  mer_readpds(file,kv=kv,/no_image) then $
                   message,'Could not read PDS label for: '+file+'!'
		if keyword_set(sequence) && (fix(info.sequence_num) ne fix(sequence)) then continue
		if keyword_set(thumbnails) && info.fullframe then continue
		if ~keyword_set(thumbnails) && info.thumbnail then continue
		if keyword_set(thumbnails) && keyword_set(ignore_focus_thumbnails) then begin
			if not mer_pds_lbl_kv_get_value(kv,'LINES',$
				lines,'INT',in_block='IMAGE', bl_type='OBJECT') then $
				message,'Could not find IMAGE LINES for: '+file
			if lines le focus_thumbnail_max_size then continue
		endif
 		if type ne 'EDR' then begin
			if not mer_pds_lbl_kv_get_value(kv,'SOFTWARE_VERSION_ID',$
					software_version,'STRING') then $
					message,'Could not find SOFTWARE_VERSION_ID for: '+file
		    if ~strcmp(software_version,current_software_version,2) then begin
				print,'Old calibration version detected!'
				print,'Skipping '+file
				continue
			endif
        endif
		files_out=[files_out,file]
	endforeach

	if keyword_set(remove_duplicates) then files_out=danika_mcam_remove_duplicate_files(files_out)

	if keyword_set(prefer_lossless) then files_out=danika_mcam_prefer_lossless_files(files_out)

	if keyword_set(resolve_multiple_pointings) then files_out=danika_mcam_resolve_multiple_pointings(files_out) else files_out=list(files_out)

	if keyword_set(resolve_multiple_filters) then files_out=danika_mcam_resolve_multiple_filters(files_out)
	
	if keyword_set(both) then begin 
		if n_elements(files_out) eq 0 then files_out = hanika_mcam_get_filelist(input_sol,type,sequence=sequence,remove_duplicates=remove_duplicates,resolve_multiple_pointings=resolve_multiple_pointings,resolve_multiple_filter=resolve_multiple_filters) $ 
			else files_out = files_out + hanika_mcam_get_filelist(input_sol,type,sequence=sequence,remove_duplicates=remove_duplicates,resolve_multiple_pointings=resolve_multiple_pointings,resolve_multiple_filter=resolve_multiple_filters)
	endif

	return,files_out
	
end
