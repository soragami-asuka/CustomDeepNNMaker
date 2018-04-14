//===========================
// NNのレイヤー設定項目フォーマットベース
//===========================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_H__

#include<guiddef.h>

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"


namespace Gravisbell {
namespace SettingData {
namespace Standard {

	/** 設定項目種別 */
	enum ItemType
	{
		ITEMTYPE_FLOAT,
		ITEMTYPE_INT,
		ITEMTYPE_STRING,
		ITEMTYPE_BOOL,
		ITEMTYPE_ENUM,
		ITEMTYPE_VECTOR2D_FLOAT,
		ITEMTYPE_VECTOR2D_INT,
		ITEMTYPE_VECTOR3D_FLOAT,
		ITEMTYPE_VECTOR3D_INT,

		ITEMTYPE_COUNT
	};

	static const U32 ITEM_NAME_MAX = 64;
	static const U32 ITEM_TEXT_MAX = 1024;
	static const U32 ITEM_ID_MAX   = 64;
	static const U32 LAYERTYPE_CODE_MAX  = 36 + 1;	/**< レイヤーの種類を識別するためのコードの最大文字数(36文字＋nullコード) */

	/** 設定項目のベース */
	class IItemBase
	{
	public:
		/** コンストラクタ */
		IItemBase(){};
		/** デストラクタ */
		virtual ~IItemBase(){};

		/** 一致演算 */
		virtual bool operator==(const IItemBase& item)const = 0;
		/** 不一致演算 */
		virtual bool operator!=(const IItemBase& item)const = 0;

		/** 自身の複製を作成する */
		virtual IItemBase* Clone()const = 0;

	public:
		/** 設定項目種別を取得する */
		virtual ItemType GetItemType()const = 0;
		/** 項目IDを取得する.
			IDはLayerConfig内において必ずユニークである.
			@param o_szIDBuf	IDを格納するバッファ.ITEM_ID_MAXのバイト数が必要.　*/
		virtual Gravisbell::ErrorCode GetConfigID(wchar_t o_szIDBuf[])const = 0;
		/** 項目名を取得する.
			@param o_szNameBuf	名前を格納するバッファ.ITEM_NAME_MAXのバイト数が必要. */
		virtual Gravisbell::ErrorCode GetConfigName(wchar_t o_szNameBuf[])const = 0;
		/** 項目の説明テキストを取得する.
			@param o_szBuf	説明文を格納するバッファ.ITEM_TEXT_MAXのバイト数が必要. */
		virtual Gravisbell::ErrorCode GetConfigText(wchar_t o_szBuf[])const = 0;

	public:
		//================================
		// ファイル保存関連.
		// 文字列本体や列挙値のIDなど構造体には保存されない細かい情報を取り扱う.
		//================================

		/** 保存に必要なバイト数を取得する */
		virtual U64 GetUseBufferByteCount()const = 0;

		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual S64 ReadFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize) = 0;

	public:
		//================================
		// 構造体を利用したデータの取り扱い.
		// 情報量が少ない代わりにアクセス速度が速い
		//================================
		/** アライメントのサイズを取得する */
		virtual U64 GetAlignmentByteCount()const = 0;

		/** 構造体に書き込む.
			@return	使用したバイト数. */
		virtual S32 WriteToStruct(BYTE* o_lpBuffer)const = 0;
		/** 構造体から読み込む.
			@return	使用したバイト数. */
		virtual S32 ReadFromStruct(const BYTE* i_lpBuffer) = 0;
	};

	/** 設定項目(実数) */
	class IItem_Float : virtual public IItemBase
	{
	public:
		/** コンストラクタ */
		IItem_Float()
			: IItemBase()
		{};
		/** デストラクタ */
		virtual ~IItem_Float(){};

	public:
		/** 値を取得する */
		virtual F32 GetValue()const = 0;
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		virtual Gravisbell::ErrorCode SetValue(F32 value) = 0;

	public:
		/** 設定可能最小値を取得する */
		virtual F32 GetMin()const = 0;
		/** 設定可能最大値を取得する */
		virtual F32 GetMax()const = 0;

		/** デフォルトの設定値を取得する */
		virtual F32 GetDefault()const = 0;
	};

	/** 設定項目(整数) */
	class IItem_Int : virtual public IItemBase
	{
	public:
		/** コンストラクタ */
		IItem_Int()
			: IItemBase()
		{};
		/** デストラクタ */
		virtual ~IItem_Int(){};

	public:
		/** 値を取得する */
		virtual S32 GetValue()const = 0;
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		virtual Gravisbell::ErrorCode SetValue(S32 value) = 0;
		
	public:
		/** 設定可能最小値を取得する */
		virtual S32 GetMin()const = 0;
		/** 設定可能最大値を取得する */
		virtual S32 GetMax()const = 0;

		/** デフォルトの設定値を取得する */
		virtual S32 GetDefault()const = 0;
	};

	/** 設定項目(文字列) */
	class IItem_String : virtual public IItemBase
	{
	public:
		/** コンストラクタ */
		IItem_String()
			: IItemBase()
		{};
		/** デストラクタ */
		virtual ~IItem_String(){};

	public:
		/** 文字列の長さを取得する */
		virtual U32 GetLength()const = 0;
		/** 値を取得する.
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual S32 GetValue(wchar_t o_szBuf[])const = 0;
		/** 値を取得する.
			@return 成功した場合文字列の先頭アドレス. */
		virtual const wchar_t* GetValue()const = 0;
		/** 値を設定する
			@param i_szBuf	設定する値
			@return 成功した場合0 */
		virtual Gravisbell::ErrorCode SetValue(const wchar_t i_szBuf[]) = 0;
		
	public:
		/** デフォルトの設定値を取得する
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual S32 GetDefault(wchar_t o_szBuf[])const = 0;
	};

	/** 設定項目(論理値) */
	class IItem_Bool : virtual public IItemBase
	{
	public:
		/** コンストラクタ */
		IItem_Bool()
			: IItemBase()
		{};
		/** デストラクタ */
		virtual ~IItem_Bool(){};

	public:
		/** 値を取得する */
		virtual bool GetValue()const = 0;
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		virtual Gravisbell::ErrorCode SetValue(bool value) = 0;

	public:
		/** デフォルトの設定値を取得する */
		virtual bool GetDefault()const = 0;
	};

	/** 設定項目(列挙型).
		※列挙要素ID=一意の英数字. 列挙要素名=利用者が認識しやすい名前.文字列.(重複可).1行. コメント=列挙要素の説明文.複数行可 */
	class IItem_Enum : virtual public IItemBase
	{
	public:
		static const U32 ID_BUFFER_MAX = 64;			/**< 列挙要素IDの最大文字数 */
		static const U32 NAME_BUFFER_MAX = 64;			/**< 列挙要素名の最大文字数 */
		static const U32 COMMENT_BUFFER_MAX = 256;		/**< コメントの最大文字数 */

	public:
		/** コンストラクタ */
		IItem_Enum()
			: IItemBase()
		{};
		/** デストラクタ */
		virtual ~IItem_Enum(){};

	public:
		/** 値を取得する */
		virtual S32 GetValue()const = 0;

		/** 値を設定する
			@param value	設定する値 */
		virtual Gravisbell::ErrorCode SetValue(S32 value) = 0;
		/** 値を設定する(文字列指定)
			@param i_szID	設定する値(文字列指定) */
		virtual Gravisbell::ErrorCode SetValue(const wchar_t i_szEnumID[]) = 0;

	public:
		/** 列挙要素数を取得する */
		virtual U32 GetEnumCount()const = 0;
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual S32 GetEnumID(S32 num, wchar_t o_szBuf[])const = 0;
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual S32 GetEnumName(S32 num, wchar_t o_szBuf[])const = 0;
		/** 列挙要素説明文を番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual S32 GetEnumText(S32 num, wchar_t o_szBuf[])const = 0;

		/** IDを指定して列挙要素番号を取得する
			@param i_szBuf　調べる列挙ID.
			@return 成功した場合0<=num<GetEnumCountの値. 失敗した場合は負の値が返る. */
		virtual S32 GetNumByID(const wchar_t i_szBuf[])const = 0;


		/** デフォルトの設定値を取得する */
		virtual S32 GetDefault()const = 0;
	};

	/** 設定項目(Vector3)(実数) */
	template<class Type>
	class IItem_Vector3D : public IItemBase
	{
	public:
		/** コンストラクタ */
		IItem_Vector3D()
			: IItemBase()
		{};
		/** デストラクタ */
		virtual ~IItem_Vector3D(){};

	public:
		/** 値を取得する */
		virtual const Vector3D<Type>& GetValue()const = 0;
		virtual Type GetValueX()const = 0;
		virtual Type GetValueY()const = 0;
		virtual Type GetValueZ()const = 0;
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		virtual Gravisbell::ErrorCode SetValue(const Vector3D<Type>& value) = 0;
		virtual Gravisbell::ErrorCode SetValueX(Type value) = 0;
		virtual Gravisbell::ErrorCode SetValueY(Type value) = 0;
		virtual Gravisbell::ErrorCode SetValueZ(Type value) = 0;

	public:
		/** 設定可能最小値を取得する */
		virtual Type GetMinX()const = 0;
		virtual Type GetMinY()const = 0;
		virtual Type GetMinZ()const = 0;
		/** 設定可能最大値を取得する */
		virtual Type GetMaxX()const = 0;
		virtual Type GetMaxY()const = 0;
		virtual Type GetMaxZ()const = 0;

		/** デフォルトの設定値を取得する */
		virtual Type GetDefaultX()const = 0;
		virtual Type GetDefaultY()const = 0;
		virtual Type GetDefaultZ()const = 0;
	};
	typedef IItem_Vector3D<F32> IItem_Vector3D_Float;
	typedef IItem_Vector3D<S32> IItem_Vector3D_Int;


	/** 設定情報 */
	class IData
	{
	public:
		/** コンストラクタ */
		IData(){}
		/** デストラクタ */
		virtual ~IData(){}

		/** 一致演算 */
		virtual bool operator==(const IData& config)const = 0;
		/** 不一致演算 */
		virtual bool operator!=(const IData& config)const = 0;

		/** 自身の複製を作成する */
		virtual IData* Clone()const = 0;

	public:
		/** レイヤー識別コードを取得する.
			@param o_guid	格納先バッファ
			@return 成功した場合0 */
		virtual Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_guid)const = 0;

	public:
		/** 設定項目数を取得する */
		virtual U32 GetItemCount()const = 0;
		/** 設定項目を番号指定で取得する */
		virtual IItemBase* GetItemByNum(U32 i_num) = 0;
		/** 設定項目を番号指定で取得する */
		virtual const IItemBase* GetItemByNum(U32 i_num)const = 0;
		/** 設定項目をID指定で取得する */
		virtual IItemBase* GetItemByID(const wchar_t i_szIDBuf[]) = 0;
		/** 設定項目をID指定で取得する */
		virtual const IItemBase* GetItemByID(const wchar_t i_szIDBuf[])const = 0;

	public:
		/** 保存に必要なバイト数を取得する */
		virtual U64 GetUseBufferByteCount()const = 0;

		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;

	public:
		/** 構造体にデータを格納する.
			@param	o_lpBuffer	構造体の先頭アドレス. 構造体はConvertNNCofigToSourceから出力されたソースを使用する. */
		virtual ErrorCode WriteToStruct(BYTE* o_lpBuffer)const = 0;
		/** 構造体からデータを読み込む.
			@param	i_lpBuffer	構造体の先頭アドレス. 構造体はConvertNNCofigToSourceから出力されたソースを使用する. */
		virtual ErrorCode ReadFromStruct(const BYTE* i_lpBuffer) = 0;
	};


}	// Standard
}	// SettingData
}	// Gravisbell



#endif	// __I_NN_LAYER_CONFIG_H__