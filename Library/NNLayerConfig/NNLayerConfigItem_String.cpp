//==================================
// 設定項目(文字列)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_String : public INNLayerConfigItem_String
	{
	private:
		std::string name;
		std::string id;

		std::string defaultValue;
		std::string value;

	public:
		/** コンストラクタ */
		NNLayerConfigItem_String(const char szName[], const char defaultValue[])
			: INNLayerConfigItem_String()
			, name (szName)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** コピーコンストラクタ */
		NNLayerConfigItem_String(const NNLayerConfigItem_String& item)
			: INNLayerConfigItem_String()
			, name (item.name)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** デストラクタ */
		virtual ~NNLayerConfigItem_String(){}
		
		/** 一致演算 */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			NNLayerConfigItem_String* pItem = (NNLayerConfigItem_String*)&item;

			if(this->name != pItem->name)
				return false;

			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** 不一致演算 */
		bool operator!=(const INNLayerConfigItemBase& item)const
		{
			return !(*this == item);
		}

		/** 自身の複製を作成する */
		INNLayerConfigItemBase* Clone()const
		{
			return new NNLayerConfigItem_String(*this);
		}
	public:
		/** 設定項目種別を取得する */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_STRING;
		}
		/** 項目名を取得する.
			@param o_szNameBuf	名前を格納するバッファ. CONFIGITEM_NAME_MAXのバイト数が必要 */
		ELayerErrorCode GetConfigName(char o_szNameBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szNameBuf, this->name.c_str(), this->name.size() + 1);

			return LAYER_ERROR_NONE;
		}
		/** 項目IDを取得する.
			@param o_szIDBuf	IDを格納するバッファ. CONFIGITEM_NAME_MAXのバイト数が必要 */
		ELayerErrorCode GetConfigID(char o_szIDBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szIDBuf, this->name.c_str(), this->name.size() + 1);

			return LAYER_ERROR_NONE;
		}

	public:
		/** 文字列の長さを取得する */
		virtual unsigned int GetLength()const
		{
			return this->name.size();
		}
		/** 値を取得する.
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual int GetValue(char o_szBuf[])const
		{
			memcpy(o_szBuf, this->value.c_str(), this->value.size() + 1);

			return 0;
		}
		/** 値を設定する
			@param i_szBuf	設定する値
			@return 成功した場合0 */
		virtual ELayerErrorCode SetValue(const char i_szBuf[])
		{
			this->value = i_szBuf;

			return LAYER_ERROR_NONE;
		}
		
	public:
		/** デフォルトの設定値を取得する
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		int GetDefault(char o_szBuf[])const
		{
			memcpy(o_szBuf, this->defaultValue.c_str(), this->defaultValue.size() + 1);

			return 0;
		}
		
	public:
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
				std::vector<char> szBuf(bufferSize+1, NULL);
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
	};
	
	/** 設定項目(実数)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_String* CreateLayerCofigItem_String(const char szName[], const char szDefaultValue[])
	{
		return new NNLayerConfigItem_String(szName, szDefaultValue);
	}
}