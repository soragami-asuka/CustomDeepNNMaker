//==================================
// 設定項目(実数)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"
#include"NNLayerConfigItemBase.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_Float : public NNLayerConfigItemBase<INNLayerConfigItem_Float>
	{
	private:
		float minValue;
		float maxValue;
		float defaultValue;

		float value;

	public:
		/** コンストラクタ */
		NNLayerConfigItem_Float(const char i_szID[], const char i_szName[], const char i_szText[], float minValue, float maxValue, float defaultValue)
			: NNLayerConfigItemBase(i_szID, i_szName, i_szText)
			, minValue(minValue)
			, maxValue(maxValue)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** コピーコンストラクタ */
		NNLayerConfigItem_Float(const NNLayerConfigItem_Float& item)
			: NNLayerConfigItemBase(item)
			, minValue(item.minValue)
			, maxValue(item.maxValue)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** デストラクタ */
		virtual ~NNLayerConfigItem_Float(){}
		
		/** 一致演算 */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const NNLayerConfigItem_Float* pItem = dynamic_cast<const NNLayerConfigItem_Float*>(&item);
			if(pItem == NULL)
				return false;

			// ベース比較
			if(NNLayerConfigItemBase::operator!=(*pItem))
				return false;

			if(this->minValue != pItem->minValue)
				return false;
			if(this->maxValue != pItem->maxValue)
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
			return new NNLayerConfigItem_Float(*this);
		}

	public:
		/** 設定項目種別を取得する */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_FLOAT;
		}

	public:
		/** 値を取得する */
		float GetValue()const
		{
			return this->value;
		}
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		ELayerErrorCode SetValue(float value)
		{
			if(value < this->minValue)
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;
			if(value > this->maxValue)
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

			this->value = value;

			return LAYER_ERROR_NONE;
		}

	public:
		/** 設定可能最小値を取得する */
		float GetMin()const
		{
			return this->minValue;
		}
		/** 設定可能最大値を取得する */
		float GetMax()const
		{
			return this->maxValue;
		}

		/** デフォルトの設定値を取得する */
		float GetDefault()const
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
			float value = *(float*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);

			this->SetValue(value);

			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;

			// 値
			*(float*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(this->value);

			return bufferPos;
		}
	};
	
	/** 設定項目(実数)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Float* CreateLayerCofigItem_Float(const char i_szID[], const char i_szName[], const char i_szText[], float minValue, float maxValue, float defaultValue)
	{
		return new NNLayerConfigItem_Float(i_szID, i_szName, i_szText, minValue, maxValue, defaultValue);
	}
}