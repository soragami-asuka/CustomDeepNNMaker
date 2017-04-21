//=======================================
// �f�[�^�t�H�[�}�b�g�̃o�C�i���`���C���^�[�t�F�[�X
//=======================================
#ifndef __GRAVISBELL_I_DATAFORMAT_BINARY_H__
#define __GRAVISBELL_I_DATAFORMAT_BINARY_H__

#include"../IDataFormatBase.h"

namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	/** �f�[�^�t�H�[�}�b�g */
	class IDataFormat : public IDataFormatBase
	{
	public:
		/** �R���X�g���N�^ */
		IDataFormat(){}
		/** �f�X�g���N�^ */
		virtual ~IDataFormat(){}

	public:
		/** ID�w��ŕϐ��̒l���擾����.(���l����t��) */
		virtual S32 GetVariableValue(const wchar_t i_szID[])const = 0;
		/** ID�w��ŕϐ��̒l���擾����.(���l����t��.)(float�^�Ƃ��Ēl��Ԃ�) */
		virtual F32 GetVariableValueAsFloat(const wchar_t i_szID[])const = 0;

	public:
		/** �f�[�^���o�C�i������ǂݍ���.
			@param	i_lpBuf		�o�C�i���f�[�^.
			@param	i_byteCount	�g�p�\�ȃo�C�g��.	
			@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
		virtual S32 LoadBinary(const BYTE i_lpBuf[], U32 i_byteCount) = 0;
	};

}	// Binary
}	// DataFormat
}	// Gravisbell



#endif // __GRAVISBELL_I_DATAFORMAT_CSV_H__