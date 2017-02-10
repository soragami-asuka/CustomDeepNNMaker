//===========================
// NN�̃��C���[�ݒ荀�ڃt�H�[�}�b�g�x�[�X
//===========================
#ifndef __I_NN_LAYER_CONFIG_H__
#define __I_NN_LAYER_CONFIG_H__

#include<guiddef.h>

#include"LayerErrorCode.h"

#ifndef BYTE
typedef unsigned char BYTE;
#endif

namespace CustomDeepNNLibrary
{
	/** �ݒ荀�ڎ�� */
	enum NNLayerConfigItemType
	{
		CONFIGITEM_TYPE_FLOAT,
		CONFIGITEM_TYPE_INT,
		CONFIGITEM_TYPE_STRING,
		CONFIGITEM_TYPE_BOOL,
		CONFIGITEM_TYPE_ENUM,

		CONFIGITEM_TYPE_COUNT
	};

	static const unsigned int CONFIGITEM_NAME_MAX = 64;
	static const unsigned int LAYERTYPE_CODE_MAX  = 36 + 1;	/**< ���C���[�̎�ނ����ʂ��邽�߂̃R�[�h�̍ő啶����(36�����{null�R�[�h) */

	/** �o�[�W�����R�[�h */
	struct VersionCode
	{
		unsigned short major;		/// ���W���[�o�[�W����	���i�����{����ύX����ꍇ�ɕύX����܂��B
		unsigned short minor;		/// �}�C�i�[�o�[�W����	�啝�Ȏd�l�ύX�E�@�\�ǉ�������ꍇ�ɕύX����܂��B
		unsigned short revision;	/// ���r�W����			�d�l�ύX�E�@�\�ǉ�������ꍇ�ɕύX����܂��B
		unsigned short build;		/// �r���h				�C���p�b�`���ƂɕύX����܂��B

		/** �R���X�g���N�^ */
		VersionCode()
			:	VersionCode(0, 0, 0, 0)
		{
		}
		/** �R���X�g���N�^ */
		VersionCode(unsigned short major, unsigned short minor, unsigned short revision, unsigned short build)
			:	major	(major)
			,	minor	(minor)
			,	revision(revision)
			,	build	(build)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		VersionCode(const VersionCode& code)
			:	VersionCode(code.major, code.minor, code.revision, code.build)
		{
		}

		/** �����Z */
		const VersionCode& operator=(const VersionCode& code)
		{
			this->major = code.major;
			this->minor = code.minor;
			this->revision = code.revision;
			this->build = code.build;

			return *this;
		}

		/** ��v���Z */
		bool operator==(const VersionCode& code)const
		{
			if(this->major != code.major)
				return false;
			if(this->minor != code.minor)
				return false;
			if(this->revision != code.revision)
				return false;
			if(this->build != code.build)
				return false;

			return true;
		}
		/** �s��v���Z */
		bool operator!=(const VersionCode& code)const
		{
			return !(*this == code);
		}
		/** ��r���Z */
		bool operator<(const VersionCode& code)const
		{
			if(this->major < code.major)
				return true;
			if(this->major > code.major)
				return false;

			if(this->minor < code.minor)
				return true;
			if(this->minor > code.minor)
				return false;

			if(this->revision < code.revision)
				return true;
			if(this->revision > code.revision)
				return false;

			if(this->build < code.build)
				return true;
			if(this->build > code.build)
				return false;

			return false;
		}
	};


	/** �ݒ荀�ڂ̃x�[�X */
	class INNLayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerConfigItemBase(){};
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfigItemBase(){};

		/** ��v���Z */
		virtual bool operator==(const INNLayerConfigItemBase& item)const = 0;
		/** �s��v���Z */
		virtual bool operator!=(const INNLayerConfigItemBase& item)const = 0;

		/** ���g�̕������쐬���� */
		virtual INNLayerConfigItemBase* Clone()const = 0;

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		virtual NNLayerConfigItemType GetItemType()const = 0;
		/** ���ږ����擾����.
			@param o_szNameBuf	���O���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̃o�C�g�����K�v */
		virtual ELayerErrorCode GetConfigName(char o_szNameBuf[])const = 0;
		/** ����ID���擾����.
			ID��LayerConfig���ɂ����ĕK�����j�[�N�ł���.
			@param o_szIDBuf	ID���i�[����o�b�t�@.CONFIGITEM_NAME_MAX�̃o�C�g�����K�v�@*/
		virtual ELayerErrorCode GetConfigID(char o_szIDBuf[])const = 0;

	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;
	};

	/** �ݒ荀��(����) */
	class INNLayerConfigItem_Float : public INNLayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerConfigItem_Float()
			: INNLayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfigItem_Float(){};

	public:
		/** �l���擾���� */
		virtual float GetValue()const = 0;
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual ELayerErrorCode SetValue(float value) = 0;

	public:
		/** �ݒ�\�ŏ��l���擾���� */
		virtual float GetMin()const = 0;
		/** �ݒ�\�ő�l���擾���� */
		virtual float GetMax()const = 0;

		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual float GetDefault()const = 0;
	};

	/** �ݒ荀��(����) */
	class INNLayerConfigItem_Int : public INNLayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerConfigItem_Int()
			: INNLayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfigItem_Int(){};

	public:
		/** �l���擾���� */
		virtual int GetValue()const = 0;
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual ELayerErrorCode SetValue(int value) = 0;
		
	public:
		/** �ݒ�\�ŏ��l���擾���� */
		virtual int GetMin()const = 0;
		/** �ݒ�\�ő�l���擾���� */
		virtual int GetMax()const = 0;

		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual int GetDefault()const = 0;
	};

	/** �ݒ荀��(������) */
	class INNLayerConfigItem_String : public INNLayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerConfigItem_String()
			: INNLayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfigItem_String(){};

	public:
		/** ������̒������擾���� */
		virtual unsigned int GetLength()const = 0;
		/** �l���擾����.
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual int GetValue(char o_szBuf[])const = 0;
		/** �l��ݒ肷��
			@param i_szBuf	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual ELayerErrorCode SetValue(const char i_szBuf[]) = 0;
		
	public:
		/** �f�t�H���g�̐ݒ�l���擾����
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual int GetDefault(char o_szBuf[])const = 0;
	};

	/** �ݒ荀��(�_���l) */
	class INNLayerConfigItem_Bool : public INNLayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerConfigItem_Bool()
			: INNLayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfigItem_Bool(){};

	public:
		/** �l���擾���� */
		virtual bool GetValue()const = 0;
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual ELayerErrorCode SetValue(bool value) = 0;

	public:
		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual bool GetDefault()const = 0;
	};

	/** �ݒ荀��(�񋓌^).
		���񋓗v�fID=��ӂ̉p����. �񋓗v�f��=���p�҂��F�����₷�����O.������.(�d����).1�s. �R�����g=�񋓗v�f�̐�����.�����s�� */
	class INNLayerConfigItem_Enum : public INNLayerConfigItemBase
	{
	public:
		static const unsigned int ID_BUFFER_MAX = 64;			/**< �񋓗v�fID�̍ő�o�C�g�� */
		static const unsigned int NAME_BUFFER_MAX = 64;			/**< �񋓗v�f���̍ő�o�C�g�� */
		static const unsigned int COMMENT_BUFFER_MAX = 256;		/**< �R�����g�̍ő�o�C�g�� */

	public:
		/** �R���X�g���N�^ */
		INNLayerConfigItem_Enum()
			: INNLayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfigItem_Enum(){};

	public:
		/** �l���擾���� */
		virtual int GetValue()const = 0;

		/** �l��ݒ肷��
			@param value	�ݒ肷��l */
		virtual ELayerErrorCode SetValue(int value) = 0;
		/** �l��ݒ肷��(������w��)
			@param i_szID	�ݒ肷��l(������w��) */
		virtual ELayerErrorCode SetValue(const char i_szEnumID[]) = 0;

	public:
		/** �񋓗v�f�����擾���� */
		virtual unsigned int GetEnumCount()const = 0;
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual int GetEnumID(int num, char o_szBuf[])const = 0;
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual int GetEnumName(int num, char o_szBuf[])const = 0;
		/** �񋓗v�f�R�����g��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual int GetEnumComment(int num, char o_szBuf[])const = 0;

		/** ID���w�肵�ė񋓗v�f�ԍ����擾����
			@param i_szBuf�@���ׂ��ID.
			@return ���������ꍇ0<=num<GetEnumCount�̒l. ���s�����ꍇ�͕��̒l���Ԃ�. */
		virtual int GetNumByID(const char i_szBuf[])const = 0;


		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual int GetDefault()const = 0;
	};



	/** �ݒ��� */
	class INNLayerConfig
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerConfig(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfig(){}

		/** ��v���Z */
		virtual bool operator==(const INNLayerConfig& config)const = 0;
		/** �s��v���Z */
		virtual bool operator!=(const INNLayerConfig& config)const = 0;

		/** ���g�̕������쐬���� */
		virtual INNLayerConfig* Clone()const = 0;

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_guid	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ELayerErrorCode GetLayerCode(GUID& o_guid)const = 0;

	public:
		/** �ݒ荀�ڐ����擾���� */
		virtual unsigned int GetItemCount()const = 0;
		/** �ݒ荀�ڂ�ԍ��w��Ŏ擾���� */
		virtual const INNLayerConfigItemBase* GetItemByNum(unsigned int i_num)const = 0;
		/** �ݒ荀�ڂ�ID�w��Ŏ擾���� */
		virtual const INNLayerConfigItemBase* GetItemByID(const char i_szIDBuf[])const = 0;

	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;
	};


}	// namespace CustomDeepNNLibrary



#endif	// __I_NN_LAYER_CONFIG_H__