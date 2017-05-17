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
	void memcpy_reverse(void* o_pDest, const void* i_pSrc, U32 i_destSize, U32 i_srcSize)
	{
		for(U32 i=0; i<min(i_destSize,i_srcSize); i++)
		{
			memcpy(&((BYTE*)o_pDest)[i_destSize-1-i], &((const BYTE*)i_pSrc)[i], 1);
		}
	}

	template<class Type>
	Type ReadBuffer(const BYTE* i_lpBuf, U32 i_size, bool i_onReverseByte)
	{
		Type buf = 0;
		if(i_onReverseByte)
		{
			memcpy_reverse(&buf, i_lpBuf, sizeof(Type), i_size);
		}
		else
		{
			memcpy(&buf, i_lpBuf, min(sizeof(Type), i_size));
		}

		return buf;
	}
	template<class Type>
	Type ReadBuffer(const BYTE* i_lpBuf, U32 i_size, DataType i_dataType, bool i_onReverseByte)
	{
		// �o�b�t�@��ǂݎ��
		switch(i_dataType)
		{
		case DataType::DATA_TYPE_SBYTE:		return (Type)ReadBuffer<S08>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_BYTE:		return (Type)ReadBuffer<U08>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_SHORT:		return (Type)ReadBuffer<S16>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_USHORT:	return (Type)ReadBuffer<U16>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_LONG:		return (Type)ReadBuffer<S32>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_ULONG:		return (Type)ReadBuffer<U32>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_LONGLONG:	return (Type)ReadBuffer<S64>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_ULONGLONG:	return (Type)ReadBuffer<U64>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_FLOAT:		return (Type)ReadBuffer<F32>(i_lpBuf, i_size, i_onReverseByte);
		case DataType::DATA_TYPE_DOUBLE:	return (Type)ReadBuffer<F64>(i_lpBuf, i_size, i_onReverseByte);
		}
		return (Type)0;
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
		if(this->dataFormat.GetOnReverseByteOrder())
		{
			memcpy_reverse(&this->m_lpBuf[0], &buf, (U32)this->m_lpBuf.size(), sizeof(buf));
		}
		else
		{
			memcpy(&this->m_lpBuf[0], &buf, this->m_lpBuf.size());
		}
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
		case DataType::DATA_TYPE_SBYTE:		this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<S08>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_BYTE:		this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<U08>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_SHORT:		this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<S16>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_USHORT:	this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<U16>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_LONG:		this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<S32>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_ULONG:		this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<U32>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_LONGLONG:	this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<S64>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_ULONGLONG:	this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<U64>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_FLOAT:		this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<F32>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		case DataType::DATA_TYPE_DOUBLE:	this->dataFormat.SetVariableValue(this->m_id.c_str(), (S32)ReadBuffer<F64>(i_lpBuf, this->m_size, this->dataFormat.GetOnReverseByteOrder()));	break;
		}

		return this->m_size;
	}



	//======================================
	// CDataItem_float.
	// float�^�A�C�e��
	//======================================
	/** �R���X�g���N�^ */
	CItem_float::CItem_float(
		CDataFormat& i_dataFormat,
		const std::wstring& i_category,
		U32 i_size, DataType i_dataType,
		const std::wstring& i_var_no, const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch)
		:	CItem_base	(i_dataFormat)
		,	m_category		(i_category)
		,	m_size			(i_size)
		,	m_dataType		(i_dataType)
		,	var_no			(i_var_no)
		,	var_x			(i_var_x)
		,	var_y			(i_var_y)
		,	var_z			(i_var_z)
		,	var_ch			(i_var_ch)
	{
	}
	/** �f�X�g���N�^ */
	CItem_float::~CItem_float()
	{
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@param	i_dataNo		�f�[�^�ԍ�.
		@param	i_lpLocalValue	���[�J���ϐ�.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_float::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		F32 value = ReadBuffer<F32>(i_lpBuf, this->m_size, this->m_dataType, this->dataFormat.GetOnReverseByteOrder());

		U32 no = this->dataFormat.GetVariableValue(this->var_no.c_str());
		U32 x  = this->dataFormat.GetVariableValue(this->var_x.c_str());
		U32 y  = this->dataFormat.GetVariableValue(this->var_y.c_str());
		U32 z  = this->dataFormat.GetVariableValue(this->var_z.c_str());
		U32 ch = this->dataFormat.GetVariableValue(this->var_ch.c_str());

		dataFormat.SetDataValue(this->m_category.c_str(), no, x, y, z, ch, value);

		return this->m_size;
	}

		
	//======================================
	// CDataItem_float_normalize_min_max.
	// float�^�A�C�e��
	//======================================
	/** �R���X�g���N�^ */
	CItem_float_normalize_min_max::CItem_float_normalize_min_max(
		CDataFormat& i_dataFormat,
		const std::wstring& i_category,
		U32 i_size, DataType i_dataType,
		const std::wstring& i_var_no,  const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch,
		const std::wstring& i_var_min, const std::wstring& i_var_max)
		:	CItem_float(i_dataFormat, i_category, i_size, i_dataType, i_var_no, i_var_x, i_var_y, i_var_z, i_var_ch)
		,	var_min	(i_var_min)
		,	var_max	(i_var_max)
	{
	}
	/** �f�X�g���N�^ */
	CItem_float_normalize_min_max::~CItem_float_normalize_min_max()
	{
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@param	i_dataNo		�f�[�^�ԍ�.
		@param	i_lpLocalValue	���[�J���ϐ�.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_float_normalize_min_max::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		F32 value = ReadBuffer<F32>(i_lpBuf, this->m_size, this->m_dataType, this->dataFormat.GetOnReverseByteOrder());
		F32 minValue = dataFormat.GetVariableValueAsFloat(this->var_min.c_str());
		F32 maxValue = dataFormat.GetVariableValueAsFloat(this->var_max.c_str());

		U32 no = this->dataFormat.GetVariableValue(this->var_no.c_str());
		U32 x  = this->dataFormat.GetVariableValue(this->var_x.c_str());
		U32 y  = this->dataFormat.GetVariableValue(this->var_y.c_str());
		U32 z  = this->dataFormat.GetVariableValue(this->var_z.c_str());
		U32 ch = this->dataFormat.GetVariableValue(this->var_ch.c_str());

		dataFormat.SetDataValueNormalize(this->m_category.c_str(), no, x, y, z, ch, (value - minValue) / (maxValue - minValue));

		return this->m_size;
	}


	//======================================
	// CDataItem_boolArray.
	// �_���l�z��^�A�C�e��
	//======================================
	/** �R���X�g���N�^ */
	CItem_boolArray::CItem_boolArray(
		CDataFormat& i_dataFormat,
		const std::wstring& i_category,
		U32 i_size, DataType i_dataType,
		const std::wstring& i_var_no, const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch)
		:	CItem_base		(i_dataFormat)
		,	m_category		(i_category)
		,	m_size			(i_size)
		,	m_dataType		(i_dataType)
		,	var_no			(i_var_no)
		,	var_x			(i_var_x)
		,	var_y			(i_var_y)
		,	var_z			(i_var_z)
		,	var_ch			(i_var_ch)
	{
	}
	/** �f�X�g���N�^ */
	CItem_boolArray::~CItem_boolArray()
	{
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@param	i_dataNo		�f�[�^�ԍ�.
		@param	i_lpLocalValue	���[�J���ϐ�.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_boolArray::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		U32 value = ReadBuffer<U32>(i_lpBuf, this->m_size, this->m_dataType, this->dataFormat.GetOnReverseByteOrder());

		U32 no = this->var_x==L"value" ? value : this->dataFormat.GetVariableValue(this->var_no.c_str());
		U32 x  = this->var_x==L"value" ? value : this->dataFormat.GetVariableValue(this->var_x.c_str());
		U32 y  = this->var_x==L"value" ? value : this->dataFormat.GetVariableValue(this->var_y.c_str());
		U32 z  = this->var_x==L"value" ? value : this->dataFormat.GetVariableValue(this->var_z.c_str());
		U32 ch = this->var_x==L"value" ? value : this->dataFormat.GetVariableValue(this->var_ch.c_str());

		dataFormat.SetDataValueNormalize(this->m_category.c_str(), no, x, y, z, ch, 1.0f);

		return this->m_size;
	}


	//======================================
	// CDataItem_items.
	// �A�C�e���z��^
	//======================================
	/** �R���X�g���N�^ */
	CItem_items::CItem_items(CDataFormat& i_dataFormat, const std::wstring& i_id, const std::wstring& i_count)
		:	CItem_items_base(i_dataFormat)
		,	m_id	(i_id)
		,	m_count	(i_count)
	{
	}
	/** �f�X�g���N�^ */
	CItem_items::~CItem_items()
	{
		auto it = this->lpItem.begin();
		while(it != this->lpItem.end())
		{
			delete *it;
			it = this->lpItem.erase(it);
		}
	}

	/** �A�C�e����ǉ����� */
	Gravisbell::ErrorCode CItem_items::AddItem(CItem_base* pItem)
	{
		this->lpItem.push_back(pItem);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@param	i_dataNo		�f�[�^�ԍ�.
		@param	i_lpLocalValue	���[�J���ϐ�.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_items::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		U32 bufPos = 0;

		S32 count = this->dataFormat.GetVariableValue(this->m_count.c_str());
		for(S32 dataNo=0; dataNo<count; dataNo++)
		{
			this->dataFormat.SetVariableValue(this->m_id.c_str(), dataNo);

			for(auto pItem : this->lpItem)
			{
				S32 useBufNum = pItem->LoadBinary(&i_lpBuf[bufPos], i_byteCount-bufPos);
				if(useBufNum < 0)
					return useBufNum;

				bufPos += useBufNum;
			}
		}

		return (S32)bufPos;
	}


	//======================================
	// CItem_data.
	// �f�[�^�^
	//======================================
	/** �R���X�g���N�^ */
	CItem_data::CItem_data(CDataFormat& i_dataFormat)
		:	CItem_items_base(i_dataFormat)
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
	Gravisbell::ErrorCode CItem_data::AddItem(CItem_base* pItem)
	{
		this->lpItem.push_back(pItem);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf			�o�C�i���擪�A�h���X.
		@param	i_byteCount		�Ǎ��\�ȃo�C�g��.
		@param	i_dataNo		�f�[�^�ԍ�.
		@param	i_lpLocalValue	���[�J���ϐ�.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CItem_data::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		U32 bufPos = 0;

		for(auto pItem : this->lpItem)
		{
			S32 useBufNum = pItem->LoadBinary(&i_lpBuf[bufPos], i_byteCount-bufPos);
			if(useBufNum < 0)
				return useBufNum;

			bufPos += useBufNum;
		}

		return (S32)bufPos;
	}


}	// Format
}	// Binary
}	// DataFormat
}	// Gravisbell
