//==========================================
// �t�H�[�}�b�g���`���邽�߂̃A�C�e��
//==========================================
#include"stdafx.h"

#include"DataFormatItem.h"
#include"DataFormatClass.h"


namespace Gravisbell {
namespace DataFormat {
namespace Binary {
namespace Format {
	
namespace
{
	template<class Type>
	Type ReadBuffer(const BYTE* i_lpBuf, U32 i_size)
	{
		Type buf = 0;
		memcpy(&buf, i_lpBuf, min(sizeof(Type), i_size));

		return buf;
	}
	template<class Type>
	Type ReadBuffer(const BYTE* i_lpBuf, U32 i_size, DataType i_dataType)
	{
		// �o�b�t�@��ǂݎ��
		switch(i_dataType)
		{
		case DataType::DATA_TYPE_SBYTE:		return (Type)ReadBuffer<S08>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_BYTE:		return (Type)ReadBuffer<U08>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_SHORT:		return (Type)ReadBuffer<S16>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_USHORT:	return (Type)ReadBuffer<U16>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_LONG:		return (Type)ReadBuffer<S32>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_ULONG:		return (Type)ReadBuffer<U32>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_LONGLONG:	return (Type)ReadBuffer<S64>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_ULONGLONG:	return (Type)ReadBuffer<U64>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_FLOAT:		return (Type)ReadBuffer<F32>(i_lpBuf, i_size);
		case DataType::DATA_TYPE_DOUBLE:	return (Type)ReadBuffer<F64>(i_lpBuf, i_size);
		}
		return (Type)0;
	}
}

	namespace Data
	{
		//======================================
		// CDataItem_float.
		// float�^�A�C�e��
		//======================================
		/** �R���X�g���N�^ */
		CDataItem_float::CDataItem_float(
			CDataFormat& i_dataFormat,
			const std::wstring& i_category,
			U32 i_size, DataType i_dataType,
			const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch)
			:	CDataItem_base	(i_dataFormat)
			,	m_category		(i_category)
			,	m_size			(i_size)
			,	m_dataType		(i_dataType)
			,	var_x			(i_var_x)
			,	var_y			(i_var_y)
			,	var_z			(i_var_z)
			,	var_ch			(i_var_ch)
		{
		}
		/** �f�X�g���N�^ */
		CDataItem_float::~CDataItem_float()
		{
		}

		/** �o�C�i���f�[�^��ǂݍ���.
			@param	i_lpBuf			�o�C�i���擪�A�h���X.
			@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
			@param	i_dataNo		�f�[�^�ԍ�.
			@param	i_lpLocalValue	���[�J���ϐ�.
			@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
		S32 CDataItem_float::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount, U32 i_dataNo)
		{
			if(i_byteCount < this->m_size)
				return -1;

			F32 value = ReadBuffer<F32>(i_lpBuf, this->m_size, this->m_dataType);

			U32 x  = this->m_dataFormat.GetVariableValue(this->var_x);
			U32 y  = this->m_dataFormat.GetVariableValue(this->var_y);
			U32 z  = this->m_dataFormat.GetVariableValue(this->var_z);
			U32 ch = this->m_dataFormat.GetVariableValue(this->var_ch);

			m_dataFormat.SetDataValue(this->m_category.c_str(), i_dataNo, x, y, z, ch, value);

			return this->m_size;
		}



	}



	//======================================
	// CItem_blank.
	// �󔒂̃A�C�e��
	//======================================
	/** �R���X�g���N�^ */
	CItem_blank::CItem_blank(CDataFormat& i_dataFormat, U32 i_size)
		:	CItem_base(i_dataFormat)
		,	m_size	(i_size)
	{
	}
	/** �f�X�g���N�^ */
	CItem_blank::~CItem_blank()
	{
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_blank::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		return m_size;
	}


	//======================================
	// CItem_signature.
	// ���ʗp�V�O�l�`���A�C�e��
	//======================================
	/** �R���X�g���N�^ */
	CItem_signature::CItem_signature(CDataFormat& i_dataFormat, U32 i_size, const std::wstring& i_buf)
		:	CItem_base	(i_dataFormat)
		,	m_size		(i_size)
	{
		// �o�b�t�@���m��
		this->m_lpBuf.resize(m_size);

		U32 buf = ConvertString2UInt(i_buf);
		memcpy(&this->m_lpBuf[0], &buf, this->m_lpBuf.size());
	}
	/** �f�X�g���N�^ */
	CItem_signature::~CItem_signature()
	{
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_signature::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		// �o�b�t�@��ǂݎ��
		std::vector<BYTE> readBuf(this->m_size);
		memcpy(&readBuf[0], i_lpBuf, this->m_size);

		// ��r
		for(U32 i=0; i<this->m_size; i++)
		{
			if(this->m_lpBuf[i] != readBuf[i])
				return -1;
		}

		return this->m_size;
	}


	//======================================
	// CItem_variable.
	// �ϐ���`�A�C�e��
	//======================================
	/** �R���X�g���N�^ */
	CItem_variable::CItem_variable(CDataFormat& i_dataFormat, U32 i_size, DataType i_dataType, const std::wstring& i_id)
		:	CItem_base	(i_dataFormat)
		,	m_size		(i_size)
		,	m_dataType	(i_dataType)
		,	m_id		(i_id)
	{
	}
	/** �f�X�g���N�^ */
	CItem_variable::~CItem_variable()
	{
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_variable::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		// �o�b�t�@��ǂݎ��
		switch(this->m_dataType)
		{
		case DataType::DATA_TYPE_SBYTE:		this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<S08>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_BYTE:		this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<U08>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_SHORT:		this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<S16>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_USHORT:	this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<U16>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_LONG:		this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<S32>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_ULONG:		this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<U32>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_LONGLONG:	this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<S64>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_ULONGLONG:	this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<U64>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_FLOAT:		this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<F32>(i_lpBuf, this->m_size));	break;
		case DataType::DATA_TYPE_DOUBLE:	this->dataFormat.SetVariableValue(this->m_id, (S32)ReadBuffer<F64>(i_lpBuf, this->m_size));	break;
		}

		return this->m_size;
	}



	//======================================
	// CItem_data.
	// �f�[�^�{�̒�`�A�C�e��
	//======================================
	/** �R���X�g���N�^ */
	CItem_data::CItem_data(CDataFormat& dataFormat, const std::wstring& i_count)
		:	CItem_base	(dataFormat)
		,	m_count		(i_count)
	{
	}
	/** �f�X�g���N�^ */
	CItem_data::~CItem_data()
	{
		auto it = this->lpItem.begin();
		while(it != this->lpItem.end())
		{
			delete *it;
			it = this->lpItem.erase(it);
		}
	}

	/** �A�C�e����ǉ����� */
	Gravisbell::ErrorCode CItem_data::AddItem(Data::CDataItem_base* pItem)
	{
		this->lpItem.push_back(pItem);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf		�o�C�i���擪�A�h���X.
		@param	i_byteCount	�Ǎ��\�ȃo�C�g��.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_data::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		U32 bufPos = 0;

		S32 count = this->dataFormat.GetVariableValue(this->m_count);
		for(S32 dataNo=0; dataNo<count; dataNo++)
		{
			for(auto pItem : this->lpItem)
			{
				S32 useBufNum = pItem->LoadBinary(&i_lpBuf[bufPos], i_byteCount-bufPos, dataNo);
				if(useBufNum < 0)
					return useBufNum;

				bufPos += useBufNum;
			}
		}

		return (S32)bufPos;
	}


}	// Format
}	// Binary
}	// DataFormat
}	// Gravisbell
