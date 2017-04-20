//====================================
// データフォーマット定義の本体情報
//====================================
#include"stdafx.h"

#include"DataFormatClass.h"


namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	/** コンストラクタ */
	CDataFormat::CDataFormat()
	:	CDataFormat	(L"", L"")
	{
	}
	/** コンストラクタ */
	CDataFormat::CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
	:	name		(i_szName)
	,	text		(i_szText)
	{
	}
	/** デストラクタ */
	CDataFormat::~CDataFormat()
	{
		this->ClearDataFormat();

		// データを全削除
		for(auto& dataInfo : this->lpData)
		{
			for(U32 i=0; i<dataInfo.second.lpData.size(); i++)
				delete[] dataInfo.second.lpData[i];
		}
		this->lpData.clear();
	}



	/** 名前の取得 */
	const wchar_t* CDataFormat::GetName()const
	{
		return name.c_str();
	}
	/** 説明文の取得 */
	const wchar_t* CDataFormat::GetText()const
	{
		return text.c_str();
	}

	/** X次元の要素数を取得 */
	U32 CDataFormat::GetBufferCountX(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_x);
	}
	/** Y次元の要素数を取得 */
	U32 CDataFormat::GetBufferCountY(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_y);
	}
	/** Z次元の要素数を取得 */
	U32 CDataFormat::GetBufferCountZ(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_z);
	}
	/** CH次元の要素数を取得 */
	U32 CDataFormat::GetBufferCountCH(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_ch);
	}

	/** データ構造を取得 */
	IODataStruct CDataFormat::GetDataStruct(const wchar_t i_szCategory[])const
	{
		return IODataStruct(this->GetBufferCountCH(i_szCategory), this->GetBufferCountX(i_szCategory), this->GetBufferCountY(i_szCategory), this->GetBufferCountZ(i_szCategory));
	}

	/** データ情報を取得 */
	const DataInfo* CDataFormat::GetDataInfo(const wchar_t i_szCategory[])const
	{
		auto it = this->lpData.find(i_szCategory);
		if(it == this->lpData.end())
			return NULL;

		return &it->second;
	}
	/** データ情報を取得 */
	DataInfo* CDataFormat::GetDataInfo(const wchar_t i_szCategory[])
	{
		auto it = this->lpData.find(i_szCategory);
		if(it == this->lpData.end())
			return NULL;

		return &it->second;
	}

	/** データ情報を追加する */
	Gravisbell::ErrorCode CDataFormat::AddDataInfo(const wchar_t i_szCategory[], const wchar_t i_x[], const wchar_t i_y[], const wchar_t i_z[], const wchar_t i_ch[], F32 i_false, F32 i_true)
	{
		if(this->lpData.count(i_szCategory) > 0)
			return ErrorCode::ERROR_CODE_COMMON_ADD_ALREADY_SAMEID;

		this->lpData[i_szCategory].dataStruct.m_x     = i_x;
		this->lpData[i_szCategory].dataStruct.m_y     = i_y;
		this->lpData[i_szCategory].dataStruct.m_z     = i_z;
		this->lpData[i_szCategory].dataStruct.m_ch    = i_ch;
		this->lpData[i_szCategory].dataStruct.m_false = i_false;
		this->lpData[i_szCategory].dataStruct.m_true  = i_true;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** データ情報に値を書き込む */
	Gravisbell::ErrorCode CDataFormat::SetDataValue(const wchar_t i_szCategory[], U32 i_no, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, F32 value)
	{
		DataInfo* pDataInfo = this->GetDataInfo(i_szCategory);
		if(pDataInfo)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		if(pDataInfo->lpData.size() <= i_no)
		{
			// バッファサイズを獲得
			U32 bufferSize = this->GetDataStruct(i_szCategory).GetDataCount();
			if(bufferSize <= 0)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// データ数が足りないためメモリを確保
			U32 pos = pDataInfo->lpData.size();

			// リサイズ
			pDataInfo->lpData.resize(i_no+1);

			// メモリ確保
			for(; pos<pDataInfo->lpData.size(); pos++)
			{
				pDataInfo->lpData[pos] = new F32[bufferSize];

				// 埋める
				for(U32 i=0; i<bufferSize; i++)
					pDataInfo->lpData[pos][i] = pDataInfo->dataStruct.m_false;
			}
		}

		// 値を設定する
		U32 size_x  = this->GetVariableValue(pDataInfo->dataStruct.m_x);
		if(i_x >= size_x)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		U32 size_y  = this->GetVariableValue(pDataInfo->dataStruct.m_y);
		if(i_y >= size_y)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		U32 size_z  = this->GetVariableValue(pDataInfo->dataStruct.m_z);
		if(i_z >= size_z)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		U32 size_ch = this->GetVariableValue(pDataInfo->dataStruct.m_ch);
		if(i_ch >= size_ch)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		U32 pos = 
			(size_y * size_x * size_ch) * i_z +
			(         size_x * size_ch) * i_y +
			(                  size_ch) * i_x +
			i_ch;

		pDataInfo->lpData[i_no][pos] = value;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** データ情報に値を書き込む */
	Gravisbell::ErrorCode CDataFormat::SetDataValue(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, bool value)
	{
		DataInfo* pDataInfo = this->GetDataInfo(i_szCategory);
		if(pDataInfo)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return this->SetDataValue(i_szCategory, i_No, i_x, i_y, i_z, i_ch, value ? pDataInfo->dataStruct.m_true : pDataInfo->dataStruct.m_false);
	}


	/** ID指定で変数の値を取得する.(直値判定付き) */
	S32 CDataFormat::GetVariableValue(const std::wstring& id)const
	{
		if(isdigit(id.c_str()[0]))
		{
			// 変数ではない
			return ConvertString2Int(id);
		}
		else
		{
			// 変数
			auto it = this->lpVariable.find(id);
			if(it == this->lpVariable.end())
				return 0;
			return it->second;
		}

		return 0;
	}
	/** ID指定で変数に値を設定する.(直値判定付き) */
	void CDataFormat::SetVariableValue(const std::wstring& id, S32 value)
	{
		this->lpVariable[id] = value;
	}
	/** 文字列を値に変換.進数判定付き */
	S32 ConvertString2Int(const std::wstring& buf)
	{
		if(buf.size() >= 2 && buf.c_str()[0] == L'0')	// 正の値
		{
			// 進数判定を行う
			if(buf.c_str()[1] == L'x')
			{
				// 16進数
				return wcstol(&buf.c_str()[2], NULL, 16);
			}
			else
			{
				// 8進数
				return wcstol(&buf.c_str()[1], NULL, 8);
			}
		}
		if(buf.size() >= 3 && buf.c_str()[0] == L'-' && buf.c_str()[1] == L'0')	// 負の値
		{
			// 進数判定を行う
			if(buf.c_str()[2] == L'x')
			{
				// 16進数
				return -wcstol(&buf.c_str()[3], NULL, 16);
			}
			else
			{
				// 8進数
				return -wcstol(&buf.c_str()[2], NULL, 8);
			}
		}
		else
		{
			return wcstol(buf.c_str(), NULL, 10);

		}
	}
	/** 文字列を値に変換.進数判定付き */
	U32 ConvertString2UInt(const std::wstring& buf)
	{
		return (U32)ConvertString2Int(buf);
	}

	/** カテゴリー数を取得する */
	U32 CDataFormat::GetCategoryCount()const
	{
		return this->lpData.size();
	}
	/** カテゴリー名を番号指定で取得する */
	const wchar_t* CDataFormat::GetCategoryNameByNum(U32 categoryNo)const
	{
		if(categoryNo >= this->lpData.size())
			return NULL;

		auto it = this->lpData.begin();
		for(U32 no=0; no<categoryNo; no++)
			it++;
		return it->first.c_str();
	}


	/** データ数を取得する */
	U32 CDataFormat::GetDataCount()const
	{
		if(this->lpData.empty())
			return 0;

		return this->lpData.begin()->second.lpData.size();;
	}

	/** データを取得する */
	const F32* CDataFormat::GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const
	{
		auto pDataInfo = this->GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return NULL;

		if(pDataInfo->lpData.size() >= i_dataNo)
			return NULL;

		return pDataInfo->lpData[i_dataNo];
	}

	/** 正規化処理.
		データの追加が終了した後、一度のみ実行. 複数回実行すると値がおかしくなるので注意. */
	Gravisbell::ErrorCode CDataFormat::Normalize()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** データフォーマット数を取得する */
	U32 CDataFormat::GetDataFormatCount()const
	{
		return this->lpDataFormat.size();
	}

	/** データフォーマットを全削除する */
	Gravisbell::ErrorCode CDataFormat::ClearDataFormat()
	{
		for(auto it : this->lpDataFormat)
		{
			delete it;
		}
		this->lpDataFormat.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** バイナリデータを読み込む.
		@param	i_lpBuf		バイナリ先頭アドレス.
		@param	i_byteCount	読込可能なバイト数.
		@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
	S32 CDataFormat::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		U32 bufPos = 0;

		for(auto pItem : this->lpDataFormat)
		{
			if(bufPos >= i_byteCount)
				return -1;

			S32 useBufNum = pItem->LoadBinary(&i_lpBuf[bufPos], i_byteCount-bufPos);
			if(useBufNum < 0)
				return useBufNum;

			bufPos += useBufNum;
		}

		return (S32)bufPos;
	}


}	// Binary
}	// DataFormat
}	// Gravisbell

