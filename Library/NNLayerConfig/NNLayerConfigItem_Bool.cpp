//==================================
// 設定項目(論理型)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"
#include"NNLayerConfigItemBase.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_Bool : virtual public NNLayerConfigItemBase<INNLayerConfigItem_Bool>
	{
	private:
		bool defaultValue;

		bool value;

	public:
		/** コンストラクタ */
		NNLayerConfigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
			: NNLayerConfigItemBase(i_szID, i_szName, i_szText)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** コピーコンストラクタ */
		NNLayerConfigItem_Bool(const NNLayerConfigItem_Bool& item)
			: NNLayerConfigItemBase(item)
			, defaultValue	(item.defaultValue)
			, value			(item.value)
		{
		}
		/** デストラクタ */
		virtual ~NNLayerConfigItem_Bool(){}
		
		/** 一致演算 */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const NNLayerConfigItem_Bool* pItem = dynamic_cast<const NNLayerConfigItem_Bool*>(&item);

			if(NNLayerConfigItemBase::operator!=(*pItem))
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
			return new NNLayerConfigItem_Bool(*this);
		}

	public:
		/** 設定項目種別を取得する */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_BOOL;
		}

	public:
		/** 値を取得する */
		bool GetValue()const
		{
			return this->value;
		}
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		ELayerErrorCode SetValue(bool value)
		{
			this->value = value;

			return LAYER_ERROR_NONE;
		}

	public:
		/** デフォルトの設定値を取得する */
		bool GetDefault()const
		{
			return this->defaultValue;
		}


	public:
		/** 保存に必要なバイト数を取得する */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int byteCount = 0;

			byteCount += sizeof(this->value);			// 値

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
			this->value = *(bool*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);


			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;

			// 値
			*(bool*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(bool);

			return bufferPos;
		}
	};
	
	/** 設定項目(実数)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Bool* CreateLayerCofigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
	{
		return new NNLayerConfigItem_Bool(i_szID, i_szName, i_szText, defaultValue);
	}
}