//==========================================
// �t�H�[�}�b�g���`���邽�߂̃A�C�e��
//==========================================
#ifndef __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_ITEM_H__
#define __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_ITEM_H__


#include<string>
#include<vector>
#include<list>
#include<set>
#include<map>

#include"Library/DataFormat/Binary.h"


namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	class CDataFormat;

	enum DataType
	{
		DATA_TYPE_SBYTE,
		DATA_TYPE_BYTE,
		DATA_TYPE_SHORT,
		DATA_TYPE_USHORT,
		DATA_TYPE_LONG,
		DATA_TYPE_ULONG,
		DATA_TYPE_LONGLONG,
		DATA_TYPE_ULONGLONG,
		DATA_TYPE_FLOAT,
		DATA_TYPE_DOUBLE,
	};

	enum ByteOrder
	{
		BYTEODER_BIG,
		BYTEODER_LITTLE,
	};

	namespace Format
	{
		class CItem_base
		{
		protected:
			CDataFormat& dataFormat;

		public:
			/** �R���X�g���N�^ */
			CItem_base(CDataFormat& i_dataFormat)
				:	dataFormat	(i_dataFormat)
			{
			}
			/** �f�X�g���N�^ */
			virtual ~CItem_base(){}

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount) = 0;
		};
		class CItem_blank : public CItem_base
		{
		private:
			U32 m_size;

		public:
			/** �R���X�g���N�^ */
			CItem_blank(CDataFormat& i_dataFormat, U32 i_size);
			/** �f�X�g���N�^ */
			virtual ~CItem_blank();
			
		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_signature : public CItem_base
		{
		private:
			U32 m_size;

			std::vector<BYTE> m_lpBuf;	/**< �V�O�l�`���Ƃ��Ďg�p����o�b�t�@.�ǂݍ��񂾏��Ƃ��̃o�b�t�@�Ƃ̔�r���s�� */

		public:
			/** �R���X�g���N�^ */
			CItem_signature(CDataFormat& i_dataFormat, U32 i_size, const std::wstring& i_buf);
			/** �f�X�g���N�^ */
			virtual ~CItem_signature();

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_variable : public CItem_base
		{
		private:
			U32 m_size;
			DataType m_dataType;
			std::wstring m_id;	/**< �ϐ�ID */

		public:
			/** �R���X�g���N�^ */
			CItem_variable(CDataFormat& i_dataFormat, U32 i_size, DataType i_dataType, const std::wstring& i_id);
			/** �f�X�g���N�^ */
			virtual ~CItem_variable();

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};

		class CItem_float : public CItem_base
		{
		private:

		protected:
			const std::wstring m_category;

			const U32 m_size;
			const DataType m_dataType;
			
			const std::wstring var_no;
			const std::wstring var_x;
			const std::wstring var_y;
			const std::wstring var_z;
			const std::wstring var_ch;

		public:
			/** �R���X�g���N�^ */
			CItem_float(
				CDataFormat& i_dataFormat,
				const std::wstring& i_category,
				U32 i_size, DataType i_dataType,
				const std::wstring& i_var_no, const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch);
			/** �f�X�g���N�^ */
			virtual ~CItem_float();

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@param	i_dataNo		�f�[�^�ԍ�.
				@param	i_lpLocalValue	���[�J���ϐ�.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_float_normalize_min_max : public CItem_float
		{
		protected:
			const std::wstring var_min;
			const std::wstring var_max;

		public:
			/** �R���X�g���N�^ */
			CItem_float_normalize_min_max(
				CDataFormat& i_dataFormat,
				const std::wstring& i_category,
				U32 i_size, DataType i_dataType,
				const std::wstring& i_var_no, const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch,
				const std::wstring& i_var_min, const std::wstring& i_var_max);
			/** �f�X�g���N�^ */
			virtual ~CItem_float_normalize_min_max();

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@param	i_dataNo		�f�[�^�ԍ�.
				@param	i_lpLocalValue	���[�J���ϐ�.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_boolArray : public CItem_base
		{
		protected:
			const std::wstring m_category;

			const U32 m_size;
			const DataType m_dataType;

			const std::wstring var_no;
			const std::wstring var_x;
			const std::wstring var_y;
			const std::wstring var_z;
			const std::wstring var_ch;

		public:
			/** �R���X�g���N�^ */
			CItem_boolArray(
				CDataFormat& i_dataFormat,
				const std::wstring& i_category,
				U32 i_size, DataType i_dataType,
				const std::wstring& i_var_no, const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch);
			/** �f�X�g���N�^ */
			virtual ~CItem_boolArray();

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@param	i_dataNo		�f�[�^�ԍ�.
				@param	i_lpLocalValue	���[�J���ϐ�.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};

		class CItem_items_base : public CItem_base
		{
		public:
			/** �R���X�g���N�^ */
			CItem_items_base(CDataFormat& i_dataFormat) : CItem_base(i_dataFormat){}
			/** �f�X�g���N�^ */
			virtual ~CItem_items_base(){}

		public:
			/** �A�C�e����ǉ����� */
			virtual Gravisbell::ErrorCode AddItem(CItem_base* pItem) = 0;
		};
		class CItem_items : public CItem_items_base
		{
		protected:
			const std::wstring m_id;
			const std::wstring m_count;

			std::list<CItem_base*> lpItem;

		public:
			/** �R���X�g���N�^ */
			CItem_items(CDataFormat& i_dataFormat, const std::wstring& i_id, const std::wstring& i_count);
			/** �f�X�g���N�^ */
			virtual ~CItem_items();

		public:
			/** �A�C�e����ǉ����� */
			Gravisbell::ErrorCode AddItem(CItem_base* pItem);

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@param	i_dataNo		�f�[�^�ԍ�.
				@param	i_lpLocalValue	���[�J���ϐ�.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_data : public CItem_items_base
		{
		protected:
			std::list<CItem_base*> lpItem;

		public:
			/** �R���X�g���N�^ */
			CItem_data(CDataFormat& i_dataFormat);
			/** �f�X�g���N�^ */
			virtual ~CItem_data();

		public:
			/** �A�C�e����ǉ����� */
			Gravisbell::ErrorCode AddItem(CItem_base* pItem);

		public:
			/** �o�C�i���f�[�^��ǂݍ���.
				@param	i_lpBuf			�o�C�i���擪�A�h���X.
				@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
				@param	i_dataNo		�f�[�^�ԍ�.
				@param	i_lpLocalValue	���[�J���ϐ�.
				@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
			virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
	}

	
}	// Binary
}	// DataFormat
}	// Gravisbell


#endif	// __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_ITEM_H__