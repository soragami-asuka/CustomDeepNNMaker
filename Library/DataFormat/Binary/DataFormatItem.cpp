//==========================================
// フォーマットを定義するためのアイテム
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
		// バッファを読み取り
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
		// float型アイテム
		//======================================
		/** コンストラクタ */
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
		/** デストラクタ */
		CDataItem_float::~CDataItem_float()
		{
		}

		/** バイナリデータを読み込む.
			@param	i_lpBuf			バイナリ先頭アドレス.
			@param	i_byteCount		読込可能なバイト数.
			@param	i_dataNo		データ番号.
			@param	i_lpLocalValue	ローカル変数.
			@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
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
	// 空白のアイテム
	//======================================
	/** コンストラクタ */
	CItem_blank::CItem_blank(CDataFormat& i_dataFormat, U32 i_size)
		:	CItem_base(i_dataFormat)
		,	m_size	(i_size)
	{
	}
	/** デストラクタ */
	CItem_blank::~CItem_blank()
	{
	}

	/** バイナリデータを読み込む.
		@param	i_lpBuf			バイナリ先頭アドレス.
		@param	i_byteCount		読込可能なバイト数.
		@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
	S32 CItem_blank::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		return m_size;
	}


	//======================================
	// CItem_signature.
	// 識別用シグネチャアイテム
	//======================================
	/** コンストラクタ */
	CItem_signature::CItem_signature(CDataFormat& i_dataFormat, U32 i_size, const std::wstring& i_buf)
		:	CItem_base	(i_dataFormat)
		,	m_size		(i_size)
	{
		// バッファを確保
		this->m_lpBuf.resize(m_size);

		U32 buf = ConvertString2UInt(i_buf);
		memcpy(&this->m_lpBuf[0], &buf, this->m_lpBuf.size());
	}
	/** デストラクタ */
	CItem_signature::~CItem_signature()
	{
	}

	/** バイナリデータを読み込む.
		@param	i_lpBuf			バイナリ先頭アドレス.
		@param	i_byteCount		読込可能なバイト数.
		@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
	S32 CItem_signature::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		// バッファを読み取り
		std::vector<BYTE> readBuf(this->m_size);
		memcpy(&readBuf[0], i_lpBuf, this->m_size);

		// 比較
		for(U32 i=0; i<this->m_size; i++)
		{
			if(this->m_lpBuf[i] != readBuf[i])
				return -1;
		}

		return this->m_size;
	}


	//======================================
	// CItem_variable.
	// 変数定義アイテム
	//======================================
	/** コンストラクタ */
	CItem_variable::CItem_variable(CDataFormat& i_dataFormat, U32 i_size, DataType i_dataType, const std::wstring& i_id)
		:	CItem_base	(i_dataFormat)
		,	m_size		(i_size)
		,	m_dataType	(i_dataType)
		,	m_id		(i_id)
	{
	}
	/** デストラクタ */
	CItem_variable::~CItem_variable()
	{
	}

	/** バイナリデータを読み込む.
		@param	i_lpBuf			バイナリ先頭アドレス.
		@param	i_byteCount		読込可能なバイト数.
		@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
	S32 CItem_variable::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		if(i_byteCount < this->m_size)
			return -1;

		// バッファを読み取り
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
	// データ本体定義アイテム
	//======================================
	/** コンストラクタ */
	CItem_data::CItem_data(CDataFormat& dataFormat, const std::wstring& i_count)
		:	CItem_base	(dataFormat)
		,	m_count		(i_count)
	{
	}
	/** デストラクタ */
	CItem_data::~CItem_data()
	{
		auto it = this->lpItem.begin();
		while(it != this->lpItem.end())
		{
			delete *it;
			it = this->lpItem.erase(it);
		}
	}

	/** アイテムを追加する */
	Gravisbell::ErrorCode CItem_data::AddItem(Data::CDataItem_base* pItem)
	{
		this->lpItem.push_back(pItem);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** バイナリデータを読み込む.
		@param	i_lpBuf		バイナリ先頭アドレス.
		@param	i_byteCount	読込可能なバイト数.
		@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
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
