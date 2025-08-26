"""
AWS S3 service for document storage and ingestion.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

# Local imports
from ..core.config import settings

logger = logging.getLogger(__name__)


class S3Service:
    """AWS S3 service for document storage and management."""
    
    def __init__(self):
        self.bucket_name = settings.s3_bucket_name
        self.region = settings.aws_region
        
        # Configure S3 client with optimizations
        config = Config(
            region_name=self.region,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50
        )
        
        # Initialize S3 client
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=self.region,
                config=config
            )
        else:
            # Use default credentials (IAM role, environment, etc.)
            self.s3_client = boto3.client('s3', region_name=self.region, config=config)
        
        self.upload_prefix = "documents/"
        self.processed_prefix = "processed/"
    
    async def generate_presigned_upload_url(
        self, 
        document_id: str, 
        filename: str, 
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for uploading documents to S3.
        
        Args:
            document_id: Unique document ID
            filename: Original filename
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned upload URL or None if error
        """
        try:
            if not self.bucket_name:
                logger.warning("S3 bucket not configured")
                return None
            
            # Create S3 key
            s3_key = f"{self.upload_prefix}{document_id}/{filename}"
            
            # Generate presigned URL
            presigned_url = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                'put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key,
                    'ContentType': self._get_content_type(filename)
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned upload URL for {filename}")
            return presigned_url
            
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None
    
    async def upload_document(
        self, 
        document_id: str, 
        filename: str, 
        file_content: bytes
    ) -> Optional[str]:
        """
        Upload document directly to S3.
        
        Args:
            document_id: Unique document ID
            filename: Original filename
            file_content: File content as bytes
            
        Returns:
            S3 key if successful, None otherwise
        """
        try:
            if not self.bucket_name:
                logger.warning("S3 bucket not configured")
                return None
            
            # Create S3 key
            s3_key = f"{self.upload_prefix}{document_id}/{filename}"
            
            # Upload to S3
            await asyncio.to_thread(
                self.s3_client.put_object,
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=self._get_content_type(filename),
                Metadata={
                    'document_id': document_id,
                    'original_filename': filename,
                    'upload_date': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Successfully uploaded {filename} to S3")
            return s3_key
            
        except Exception as e:
            logger.error(f"Error uploading document to S3: {str(e)}")
            return None
    
    async def download_document(self, s3_key: str) -> Optional[bytes]:
        """
        Download document from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            File content as bytes or None if error
        """
        try:
            if not self.bucket_name:
                logger.warning("S3 bucket not configured")
                return None
            
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            content = response['Body'].read()
            logger.info(f"Downloaded document from S3: {s3_key}")
            return content
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.warning(f"Document not found in S3: {s3_key}")
            else:
                logger.error(f"Error downloading from S3: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error downloading document from S3: {str(e)}")
            return None
    
    async def copy_to_processed(self, s3_key: str, document_id: str) -> Optional[str]:
        """
        Copy document to processed folder after successful processing.
        
        Args:
            s3_key: Original S3 key
            document_id: Document ID
            
        Returns:
            New S3 key in processed folder or None if error
        """
        try:
            if not self.bucket_name:
                return None
            
            filename = os.path.basename(s3_key)
            processed_key = f"{self.processed_prefix}{document_id}/{filename}"
            
            # Copy object
            await asyncio.to_thread(
                self.s3_client.copy_object,
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': s3_key},
                Key=processed_key,
                MetadataDirective='COPY'
            )
            
            logger.info(f"Copied document to processed folder: {processed_key}")
            return processed_key
            
        except Exception as e:
            logger.error(f"Error copying to processed folder: {str(e)}")
            return None
    
    async def delete_document(self, s3_key: str) -> bool:
        """
        Delete document from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.bucket_name:
                return False
            
            await asyncio.to_thread(
                self.s3_client.delete_object,
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"Deleted document from S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document from S3: {str(e)}")
            return False
    
    async def list_documents(
        self, 
        prefix: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List documents in S3 bucket.
        
        Args:
            prefix: S3 key prefix to filter by
            max_keys: Maximum number of keys to return
            
        Returns:
            List of document metadata
        """
        try:
            if not self.bucket_name:
                return []
            
            list_params = {
                'Bucket': self.bucket_name,
                'MaxKeys': max_keys
            }
            
            if prefix:
                list_params['Prefix'] = prefix
            
            response = await asyncio.to_thread(
                self.s3_client.list_objects_v2,
                **list_params
            )
            
            documents = []
            for obj in response.get('Contents', []):
                documents.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            logger.info(f"Listed {len(documents)} documents from S3")
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents from S3: {str(e)}")
            return []
    
    async def get_document_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Document metadata or None if error
        """
        try:
            if not self.bucket_name:
                return None
            
            response = await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            metadata = {
                'content_length': response['ContentLength'],
                'content_type': response.get('ContentType', ''),
                'last_modified': response['LastModified'],
                'etag': response['ETag'],
                'metadata': response.get('Metadata', {})
            }
            
            return metadata
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.warning(f"Document not found in S3: {s3_key}")
            else:
                logger.error(f"Error getting metadata from S3: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting document metadata: {str(e)}")
            return None
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension."""
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.txt': 'text/plain',
            '.md': 'text/markdown'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    async def create_bucket_if_not_exists(self) -> bool:
        """
        Create S3 bucket if it doesn't exist.
        
        Returns:
            True if bucket exists or was created, False otherwise
        """
        try:
            if not self.bucket_name:
                logger.error("S3 bucket name not configured")
                return False
            
            # Check if bucket exists
            try:
                await asyncio.to_thread(
                    self.s3_client.head_bucket,
                    Bucket=self.bucket_name
                )
                logger.info(f"S3 bucket {self.bucket_name} already exists")
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code != '404':
                    logger.error(f"Error checking bucket: {str(e)}")
                    return False
            
            # Create bucket
            create_params = {'Bucket': self.bucket_name}
            
            # Add LocationConstraint for regions other than us-east-1
            if self.region != 'us-east-1':
                create_params['CreateBucketConfiguration'] = {
                    'LocationConstraint': self.region
                }
            
            await asyncio.to_thread(
                self.s3_client.create_bucket,
                **create_params
            )
            
            # Set bucket versioning
            await asyncio.to_thread(
                self.s3_client.put_bucket_versioning,
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            logger.info(f"Created S3 bucket: {self.bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating S3 bucket: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on S3 service.
        
        Returns:
            Health check results
        """
        try:
            if not self.bucket_name:
                return {
                    "status": "unhealthy",
                    "error": "S3 bucket not configured",
                    "bucket_accessible": False
                }
            
            # Check bucket access
            bucket_accessible = False
            try:
                await asyncio.to_thread(
                    self.s3_client.head_bucket,
                    Bucket=self.bucket_name
                )
                bucket_accessible = True
            except Exception as e:
                logger.warning(f"Bucket access check failed: {str(e)}")
            
            # Test list operation
            list_test = False
            try:
                await self.list_documents(max_keys=1)
                list_test = True
            except Exception as e:
                logger.warning(f"List operation test failed: {str(e)}")
            
            return {
                "status": "healthy" if bucket_accessible and list_test else "unhealthy",
                "bucket_name": self.bucket_name,
                "region": self.region,
                "bucket_accessible": bucket_accessible,
                "list_operation": list_test
            }
            
        except Exception as e:
            logger.error(f"S3 health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            } 