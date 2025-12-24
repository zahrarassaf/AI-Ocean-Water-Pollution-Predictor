"""
Google Drive integration for data loading
"""

import os
import io
import gdown
import zipfile
from pathlib import Path
from typing import List, Optional, Union
import logging
import pandas as pd
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from ..config.constants import DATA_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class GoogleDriveLoader:
    """Load data from Google Drive"""
    
    # Scopes for Google Drive API
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Google Drive loader
        
        Parameters
        ----------
        credentials_path : str, optional
            Path to Google OAuth credentials JSON file
        """
        self.credentials_path = credentials_path
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None
        
        # Token file stores user's access and refresh tokens
        token_file = Path('token.json')
        
        if token_file.exists():
            creds = Credentials.from_authorized_user_file(str(token_file), self.SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if self.credentials_path:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.SCOPES
                    )
                else:
                    # Try to find credentials in standard locations
                    possible_paths = [
                        'credentials.json',
                        'client_secrets.json',
                        '.credentials/google_credentials.json'
                    ]
                    
                    for path in possible_paths:
                        if Path(path).exists():
                            flow = InstalledAppFlow.from_client_secrets_file(
                                path, self.SCOPES
                            )
                            break
                    else:
                        raise FileNotFoundError(
                            "Google OAuth credentials not found. "
                            "Please provide credentials_path or place credentials.json in project root."
                        )
                
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Authenticated with Google Drive API")
    
    def download_file_by_id(self, 
                          file_id: str, 
                          output_path: Union[str, Path],
                          mime_type: Optional[str] = None) -> Path:
        """
        Download file from Google Drive by file ID
        
        Parameters
        ----------
        file_id : str
            Google Drive file ID
        output_path : str or Path
            Output file path
        mime_type : str, optional
            MIME type of the file
            
        Returns
        -------
        Path
            Path to downloaded file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            
            with open(output_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.info(f"Downloaded {int(status.progress() * 100)}%")
            
            logger.info(f"Downloaded file to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            raise
    
    def download_folder(self, 
                       folder_id: str, 
                       output_dir: Union[str, Path]) -> List[Path]:
        """
        Download all files from a Google Drive folder
        
        Parameters
        ----------
        folder_id : str
            Google Drive folder ID
        output_dir : str or Path
            Output directory
            
        Returns
        -------
        List[Path]
            List of downloaded file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        try:
            # List all files in the folder
            query = f"'{folder_id}' in parents and trashed = false"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType)"
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                logger.warning(f"No files found in folder {folder_id}")
                return downloaded_files
            
            logger.info(f"Found {len(files)} files in folder")
            
            for file in files:
                file_id = file['id']
                file_name = file['name']
                mime_type = file.get('mimeType')
                
                output_path = output_dir / file_name
                
                logger.info(f"Downloading {file_name}...")
                downloaded_path = self.download_file_by_id(
                    file_id, output_path, mime_type
                )
                downloaded_files.append(downloaded_path)
            
            logger.info(f"Downloaded {len(downloaded_files)} files to {output_dir}")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading folder {folder_id}: {e}")
            raise
    
    def download_from_shareable_link(self, 
                                   link: str, 
                                   output_path: Union[str, Path]) -> Path:
        """
        Download file from shareable Google Drive link using gdown
        
        Parameters
        ----------
        link : str
            Google Drive shareable link
        output_path : str or Path
            Output file path
            
        Returns
        -------
        Path
            Path to downloaded file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract file ID from link
            file_id = self._extract_file_id_from_link(link)
            
            # Download using gdown
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                str(output_path),
                quiet=False
            )
            
            logger.info(f"Downloaded file from link to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading from link {link}: {e}")
            raise
    
    def _extract_file_id_from_link(self, link: str) -> str:
        """Extract file ID from Google Drive shareable link"""
        import re
        
        # Pattern for Google Drive links
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'open\?id=([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, link)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract file ID from link: {link}")
    
    def download_all_project_files(self, 
                                 file_links: List[str],
                                 output_dir: Union[str, Path] = DATA_DIR / 'raw') -> Dict[str, Path]:
        """
        Download all project files from Google Drive links
        
        Parameters
        ----------
        file_links : list
            List of Google Drive shareable links
        output_dir : str or Path
            Output directory
            
        Returns
        -------
        dict
            Dictionary mapping file names to paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = {}
        
        for i, link in enumerate(file_links):
            try:
                # Generate filename
                if i < len(file_links):
                    filename = f"dataset_{i+1}.csv"
                else:
                    filename = f"dataset_{link[-10:]}.csv"
                
                output_path = output_dir / filename
                
                logger.info(f"Downloading file {i+1}/{len(file_links)}...")
                downloaded_path = self.download_from_shareable_link(link, output_path)
                
                # Store mapping
                downloaded_files[filename] = downloaded_path
                
            except Exception as e:
                logger.error(f"Failed to download file {i+1}: {e}")
                continue
        
        logger.info(f"Downloaded {len(downloaded_files)}/{len(file_links)} files")
        return downloaded_files
    
    def extract_zip_files(self, 
                         input_dir: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Extract all ZIP files in directory
        
        Parameters
        ----------
        input_dir : str or Path
            Directory containing ZIP files
        output_dir : str or Path, optional
            Output directory for extracted files
            
        Returns
        -------
        List[Path]
            List of extracted file paths
        """
        input_dir = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_dir.parent / "extracted"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        
        # Find all ZIP files
        zip_files = list(input_dir.glob("*.zip"))
        
        if not zip_files:
            logger.info(f"No ZIP files found in {input_dir}")
            return extracted_files
        
        for zip_file in zip_files:
            try:
                logger.info(f"Extracting {zip_file.name}...")
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extract all files
                    zip_ref.extractall(output_dir)
                    
                    # Get list of extracted files
                    extracted = [output_dir / name for name in zip_ref.namelist()]
                    extracted_files.extend(extracted)
                
                logger.info(f"Extracted {len(extracted)} files from {zip_file.name}")
                
            except Exception as e:
                logger.error(f"Error extracting {zip_file.name}: {e}")
                continue
        
        logger.info(f"Extracted total {len(extracted_files)} files to {output_dir}")
        return extracted_files

# Simplified version using gdown (no OAuth required)
class SimpleDriveDownloader:
    """Simplified Google Drive downloader using gdown only"""
    
    def __init__(self):
        logger.info("Initialized SimpleDriveDownloader")
    
    def download_files(self, 
                      file_links: List[str],
                      output_dir: Union[str, Path] = DATA_DIR / 'raw',
                      filenames: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Download files from Google Drive links
        
        Parameters
        ----------
        file_links : list
            List of Google Drive shareable links
        output_dir : str or Path
            Output directory
        filenames : list, optional
            Custom filenames for downloaded files
            
        Returns
        -------
        dict
            Dictionary mapping original filenames to downloaded paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = {}
        
        for i, link in enumerate(file_links):
            try:
                # Generate filename
                if filenames and i < len(filenames):
                    filename = filenames[i]
                else:
                    # Try to get filename from link or use generic name
                    filename = self._get_filename_from_link(link) or f"dataset_{i+1}.csv"
                
                # Ensure filename has extension
                if not any(filename.endswith(ext) for ext in ['.csv', '.nc', '.h5', '.json']):
                    filename += '.csv'
                
                output_path = output_dir / filename
                
                logger.info(f"Downloading {filename} ({i+1}/{len(file_links)})...")
                
                # Download file
                self._download_with_gdown(link, output_path)
                
                downloaded_files[filename] = output_path
                logger.info(f"✓ Downloaded {filename}")
                
            except Exception as e:
                logger.error(f"✗ Failed to download file {i+1}: {e}")
                continue
        
        logger.info(f"Download completed: {len(downloaded_files)}/{len(file_links)} files")
        return downloaded_files
    
    def _get_filename_from_link(self, link: str) -> Optional[str]:
        """Extract filename from Google Drive link if possible"""
        import re
        
        # Check if filename is in URL parameters
        match = re.search(r'[?&]name=([^&]+)', link)
        if match:
            return match.group(1)
        
        return None
    
    def _download_with_gdown(self, link: str, output_path: Path):
        """Download file using gdown with proper file ID extraction"""
        import gdown
        
        # Extract file ID from link
        file_id = self._extract_file_id(link)
        
        if not file_id:
            raise ValueError(f"Could not extract file ID from link: {link}")
        
        # Construct direct download URL
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        # Download file
        gdown.download(
            download_url,
            str(output_path),
            quiet=False,
            fuzzy=True
        )
        
        # Verify file was downloaded
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise IOError(f"Downloaded file is empty or doesn't exist: {output_path}")
    
    def _extract_file_id(self, link: str) -> Optional[str]:
        """Extract file ID from various Google Drive link formats"""
        import re
        
        patterns = [
            # Standard view link
            r'/file/d/([a-zA-Z0-9_-]+)',
            # Open link
            r'open\?id=([a-zA-Z0-9_-]+)',
            # Direct ID
            r'id=([a-zA-Z0-9_-]+)',
            # Shortened URL (needs following redirect)
            r'/d/([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, link)
            if match:
                return match.group(1)
        
        return None
