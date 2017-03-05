//==================================
// 設定項目(文字列)
//==================================
#include "stdafx.h"

#include"LayerConfig.h"
#include"LayerConfigItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace NeuralNetwork {

	class LayerConfigItem_String : public LayerConfigItemBase<ILayerConfigItem_String>
	{
	private:
		std::wstring defaultValue;
		std::wstring value;

	public:
		/** コンストラクタ */
		LayerConfigItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t defaultValue[])
			: LayerConfigItemBase(i_szID, i_szName, i_szText)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** コピーコンストラクタ */
		LayerConfigItem_String(const LayerConfigItem_String& item)
			: LayerConfigItemBase(item)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** デストラクタ */
		virtual ~LayerConfigItem_String(){}
		
		/** 一致演算 */
		bool operator==(const ILayerConfigItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const LayerConfigItem_String* pItem = dynamic_cast<const LayerConfigItem_String*>(&item);
			if(pItem == NULL)
				return false;

			// ベース比較
			if(LayerConfigItemBase::operator!=(*pItem))
				return false;

			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** 不一致演算 */
		bool operator!=(const ILayerConfigItemBase& item)const
		{
			return !(*this == item);
		}

		/** 自身の複製を作成する */
		ILayerConfigItemBase* Clone()const
		{
			return new LayerConfigItem_String(*this);
		}
	public:
		/** 設定項目種別を取得する */
		LayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_STRING;
		}

	public:
		/** 文字列の長さを取得する */
		virtual unsigned int GetLength()const
		{
			return this->value.size();
		}
		/** 値を取得する.
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual int GetValue(wchar_t o_szBuf[])const
		{
			wcscpy(o_szBuf, this->value.c_str());

			return 0;
		}
		/** 値を取得する.
			@return 成功した場合文字列の先頭アドレス. */
		virtual const wchar_t* GetValue()const
		{
			return this->value.c_str();
		}
		/** 値を設定する
			@param i_szBuf	設定する値
			@return 成功した場合0 */
		virtual ErrorCode SetValue(const wchar_t i_szBuf[])
		{
			this->value = i_szBuf;

			return ERROR_CODE_NONE;
		}
		
	public:
		/** デフォルトの設定値を取得する
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		int GetDefault(wchar_t o_szBuf[])const
		{
			wcscpy(o_szBuf, this->defaultValue.c_str());

			return 0;
		}
		
	public:
		//================================
		// ファイル保存関連.
		// 文字列本体や列挙値のIDなど構造体には保存されない細かい情報を取り扱う.
		//================================

		/** 保存に必要なバイト数を取得する */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int byteCount = 0;

			byteCount += sizeof(unsigned int);		// 値のバッファサイズ
			byteCount += this->value.size();		// 値

			return byteCount;
		}

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
		{
			if(i_bufferSize < (int)this->GetUseBufferByteCount())
				return -1;

			unsigned int bufferPos = 0;

			// 値
			{
				// バッファサイズ
				unsigned int bufferSize = *(unsigned int*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(unsigned int);

				// 値
				std::vector<wchar_t> szBuf(bufferSize+1, NULL);
				for(unsigned int i=0; i<bufferSize; i++)
				{
					szBuf[i] = i_lpBuffer[bufferPos++];
				}
				this->value = &szBuf[0];
			}


			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// 値
			{
				// バッファサイズ
				unsigned int bufferSize = this->value.size();;
				*(unsigned int*)&o_lpBuffer[bufferPos] = bufferSize;
				bufferPos += sizeof(unsigned int);

				// 値
				memcpy(&o_lpBuffer[bufferPos], this->value.c_str(), bufferSize);
				bufferPos += bufferSize;
			}


			return bufferPos;
		}
		
	public:
		//================================
		// 構造体を利用したデータの取り扱い.
		// 情報量が少ない代わりにアクセス速度が速い
		//================================

		/** 構造体に書き込む.
			@return	使用したバイト数. */
		S32 WriteToStruct(BYTE* o_lpBuffer)const
		{
			const wchar_t* value = this->GetValue();

			*(const wchar_t**)o_lpBuffer = value;

			return sizeof(const wchar_t*);
		}
		/** 構造体から読み込む.
			@return	使用したバイト数. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			const wchar_t* value = *(const wchar_t**)i_lpBuffer;

			this->SetValue(value);

			return sizeof(const wchar_t*);
		}
	};
	
	/** 設定項目(実数)を作成する */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_String* CreateLayerCofigItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t szDefaultValue[])
	{
		return new LayerConfigItem_String(i_szID, i_szName, i_szText, szDefaultValue);
	}

}	// NeuralNetwork
}	// Gravisbell